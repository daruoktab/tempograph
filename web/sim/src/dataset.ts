export interface UserBlock {
  name: string;
  age?: number;
  occupation?: string;
  location?: string;
  traits?: string[];
  backstory?: string;
}

export interface SecondaryPersona {
  name: string;
  relationship: string;
  traits?: string[];
}

export interface Turn {
  speaker: string;
  text: string;
}

export interface GtEntity {
  name: string;
  type: string;
  context: string;
}

export interface GroundTruth {
  turn_id: number;
  session_id?: number;
  speaker?: string;
  entities_mentioned?: GtEntity[];
}

export interface Session {
  session_id: number;
  date: string;
  datetime?: string;
  turns: Turn[];
  summary?: string;
  ground_truths?: GroundTruth[];
}

export interface SimDataset {
  user: UserBlock;
  secondary_personas: SecondaryPersona[];
  sessions: Session[];
}

export interface LifeEvent {
  id: string;
  date: string;
  description: string;
  caused_by: string[];
}
