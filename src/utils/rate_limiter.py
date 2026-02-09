# src/utils/rate_limiter.py
"""
Rate Limiter untuk Gemini API
=============================
Mengatur rate limiting per model untuk menghindari API quota errors.
"""

import asyncio
import time
import logging
from typing import Dict, Optional
from dataclasses import dataclass, field
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class RateLimiterState:
    """State tracking for a single model's rate limiter"""
    model_name: str
    rpm: int
    tpm: int
    rpd: Optional[int]
    min_delay: float
    
    # Request tracking
    request_times: deque = field(default_factory=lambda: deque(maxlen=1000))
    token_counts: deque = field(default_factory=lambda: deque(maxlen=1000))
    
    # Daily request tracking (for RPD limit)
    daily_requests: int = 0
    daily_reset_time: float = field(default_factory=time.time)
    
    # Adaptive delay
    last_request_time: float = 0.0
    consecutive_rate_limits: int = 0
    current_delay: float = 0.0
    
    def __post_init__(self):
        """Initialize current delay based on rate limit"""
        self.current_delay = self.min_delay


class RateLimiter:
    """
    Rate limiter untuk Gemini API dengan dukungan:
    - Per-model rate limiting (RPM, TPM, RPD)
    - Adaptive delay berdasarkan response
    - Exponential backoff pada rate limit errors
    - Async support
    """
    
    def __init__(self, enabled: bool = True):
        self._states: Dict[str, RateLimiterState] = {}
        self._lock = asyncio.Lock()
        self._enabled = enabled
        self._max_retry_delay = 60.0
        self._backoff_multiplier = 2.0
        
    def _get_state(self, model_name: str) -> RateLimiterState:
        """Get or create rate limiter state for a model"""
        if model_name not in self._states:
            # Import here to avoid circular imports
            from ..config import get_rate_limit
            
            rate_limit = get_rate_limit(model_name)
            self._states[model_name] = RateLimiterState(
                model_name=model_name,
                rpm=rate_limit.rpm,
                tpm=rate_limit.tpm,
                rpd=rate_limit.rpd,
                min_delay=rate_limit.min_delay_seconds
            )
            logger.debug(
                f"Created rate limiter for {model_name}: "
                f"RPM={rate_limit.rpm}, TPM={rate_limit.tpm}, RPD={rate_limit.rpd}, "
                f"min_delay={rate_limit.min_delay_seconds:.3f}s"
            )
        return self._states[model_name]
    
    def _check_daily_reset(self, state: RateLimiterState):
        """Reset daily request count if a new day has started"""
        current_time = time.time()
        # Reset if more than 24 hours since last reset
        if current_time - state.daily_reset_time >= 86400:  # 24 hours
            state.daily_requests = 0
            state.daily_reset_time = current_time
            logger.info(f"Daily request count reset for {state.model_name}")
    
    def _calculate_rpm_delay(self, state: RateLimiterState) -> float:
        """Calculate delay needed to stay within RPM limit"""
        current_time = time.time()
        window_start = current_time - 60  # 1 minute window
        
        # Count requests in the last minute
        recent_requests = sum(
            1 for t in state.request_times 
            if t > window_start
        )
        
        if recent_requests >= state.rpm:
            # Find when the oldest request in window will expire
            oldest_in_window = min(
                (t for t in state.request_times if t > window_start),
                default=current_time
            )
            wait_time = oldest_in_window + 60 - current_time
            return max(0, wait_time)
        
        return 0
    
    def _calculate_tpm_delay(self, state: RateLimiterState) -> float:
        """Calculate delay needed to stay within TPM limit"""
        current_time = time.time()
        window_start = current_time - 60  # 1 minute window
        
        # Sum tokens in the last minute
        recent_tokens = 0
        for i, t in enumerate(state.request_times):
            if t > window_start and i < len(state.token_counts):
                recent_tokens += state.token_counts[i]
        
        if recent_tokens >= state.tpm * 0.9:  # 90% threshold
            # Wait until some tokens "expire" from the window
            return state.min_delay * 2
        
        return 0
    
    async def wait_if_needed(
        self, 
        model_name: str, 
        estimated_tokens: int = 1000
    ) -> float:
        """
        Wait if necessary to comply with rate limits.
        
        Args:
            model_name: Name of the model being called
            estimated_tokens: Estimated tokens for this request
            
        Returns:
            Actual wait time in seconds
        """
        if not self._enabled:
            return 0.0
        
        async with self._lock:
            state = self._get_state(model_name)
            self._check_daily_reset(state)
            
            # Check RPD (Requests Per Day) limit
            if state.rpd is not None:
                if state.daily_requests >= state.rpd:
                    logger.error(
                        f"Daily request limit REACHED for {model_name}: "
                        f"{state.daily_requests}/{state.rpd}. "
                        f"Cannot make more requests today!"
                    )
                    raise RuntimeError(
                        f"Daily request limit reached for {model_name}. "
                        f"Used {state.daily_requests}/{state.rpd} requests."
                    )
                elif state.daily_requests >= state.rpd * 0.9:
                    logger.warning(
                        f"Daily request limit approaching for {model_name}: "
                        f"{state.daily_requests}/{state.rpd} (90%+ used)"
                    )
            
            # Calculate required delays
            rpm_delay = self._calculate_rpm_delay(state)
            tpm_delay = self._calculate_tpm_delay(state)
            min_delay = state.current_delay
            
            # Time since last request
            time_since_last = time.time() - state.last_request_time
            remaining_min_delay = max(0, min_delay - time_since_last)
            
            # Take the maximum of all delays
            total_delay = max(rpm_delay, tpm_delay, remaining_min_delay)
            
            if total_delay > 0:
                logger.debug(
                    f"Rate limiting {model_name}: waiting {total_delay:.2f}s "
                    f"(rpm_delay={rpm_delay:.2f}, tpm_delay={tpm_delay:.2f}, "
                    f"min_delay={remaining_min_delay:.2f})"
                )
                await asyncio.sleep(total_delay)
            
            # Record this request
            current_time = time.time()
            state.request_times.append(current_time)
            state.token_counts.append(estimated_tokens)
            state.last_request_time = current_time
            
            return total_delay
    
    def record_success(self, model_name: str, actual_tokens: int = 0):
        """Record a successful API call"""
        state = self._get_state(model_name)
        
        # Update token count if we have actual data
        if actual_tokens > 0 and state.token_counts:
            state.token_counts[-1] = actual_tokens
        
        # Increment daily request count
        state.daily_requests += 1
        
        # Reset consecutive rate limits on success
        if state.consecutive_rate_limits > 0:
            state.consecutive_rate_limits = 0
            # Gradually reduce delay back to minimum
            state.current_delay = max(
                state.min_delay,
                state.current_delay / self._backoff_multiplier
            )
    
    def record_rate_limit_error(self, model_name: str):
        """Record a rate limit error and increase delay"""
        state = self._get_state(model_name)
        state.consecutive_rate_limits += 1
        
        # Exponential backoff
        state.current_delay = min(
            self._max_retry_delay,
            state.current_delay * self._backoff_multiplier
        )
        
        logger.warning(
            f"Rate limit hit for {model_name}. "
            f"Consecutive: {state.consecutive_rate_limits}, "
            f"New delay: {state.current_delay:.2f}s"
        )
    
    def get_stats(self, model_name: str) -> Dict:
        """Get current rate limiter statistics for a model"""
        state = self._get_state(model_name)
        current_time = time.time()
        window_start = current_time - 60
        
        recent_requests = sum(1 for t in state.request_times if t > window_start)
        recent_tokens = sum(
            state.token_counts[i] 
            for i, t in enumerate(state.request_times) 
            if t > window_start and i < len(state.token_counts)
        )
        
        return {
            "model": model_name,
            "rpm_used": recent_requests,
            "rpm_limit": state.rpm,
            "tpm_used": recent_tokens,
            "tpm_limit": state.tpm,
            "rpd_used": state.daily_requests,
            "rpd_limit": state.rpd,
            "current_delay": state.current_delay,
            "consecutive_rate_limits": state.consecutive_rate_limits
        }


# Global rate limiter instance
_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    """Get or create global rate limiter instance"""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
    return _rate_limiter


def reset_rate_limiter():
    """Reset global rate limiter (useful for testing)"""
    global _rate_limiter
    _rate_limiter = None


# Synchronous wrapper for non-async code
def sync_wait_if_needed(model_name: str, estimated_tokens: int = 1000) -> float:
    """Synchronous version of wait_if_needed"""
    limiter = get_rate_limiter()
    
    # Run in event loop
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If we're already in an async context, just calculate delay
            state = limiter._get_state(model_name)
            delay = state.current_delay
            time.sleep(delay)
            return delay
        else:
            return loop.run_until_complete(
                limiter.wait_if_needed(model_name, estimated_tokens)
            )
    except RuntimeError:
        # No event loop, create one
        return asyncio.run(limiter.wait_if_needed(model_name, estimated_tokens))

