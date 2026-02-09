import argparse
import json
import logging
import os
import random
import re
import sys
import time
from datetime import date, datetime, timedelta

sys.path.insert(0, ".")
from src.utils import run_gemini, set_gemini_key, set_token_log_path, log_token_usage

# Import for structured outputs and safety settings
from google.generativeai.generative_models import GenerativeModel
from google.generativeai import caching
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Rate limiting for ground truth generation
REQUESTS_PER_MINUTE = 6
DELAY_BETWEEN_REQUESTS = 60 / REQUESTS_PER_MINUTE  # 10 seconds

# --- Prompts for AI Generation ---


PERSONA_GENERATION_PROMPT = """
Anda adalah penulis karakter fiksi yang ahli. Tugas Anda adalah membuat profil pengguna (User Persona) yang realistis dan mendalam untuk simulasi percakapan sehari-hari dalam Bahasa Indonesia.

Kriteria Karakter:
1.  **Kewarganegaraan**: Indonesia (WNI).
2.  **Umur**: Dewasa muda atau dewasa (20-40 tahun).
3.  **Pekerjaan**: Spesifik dan relatable (misal: Desainer Grafis, Akuntan, Barista, Mahasiswa S2, dll).
4.  **Lokasi**: Salah satu kota besar di Indonesia (Jakarta, Bandung, Surabaya, Yogyakarta, dll).
5.  **Kepribadian**: Kompleks, punya kelebihan dan kekurangan.
6.  **Gaya Bicara**: Natural, menggunakan bahasa sehari-hari (bisa "aku-kamu" atau "gue-lo" tergantung daerah/umur).

Output HARUS berupa JSON valid dengan struktur berikut:
{
  "user": {
    "name": "Nama Lengkap",
    "age": 25,
    "occupation": "Pekerjaan",
    "location": "Kota, Indonesia",
    "traits": ["Sifat 1", "Sifat 2", "Sifat 3"],
    "hobbies": ["Hobi 1", "Hobi 2"],
    "interests": ["Minat 1", "Minat 2"],
    "conversation_topics": ["Topik 1", "Topik 2"],
    "backstory": "Deskripsi singkat latar belakang kehidupan, keluarga, dan situasi saat ini (2-3 kalimat)."
  },
  "secondary_personas": [
    {
      "name": "Nama Teman/Keluarga 1",
      "relationship": "Hubungan (misal: Sahabat, Pacar, Kakak)",
      "traits": ["Sifat 1", "Sifat 2"]
    },
    {
      "name": "Nama Teman/Keluarga 2",
      "relationship": "Hubungan",
      "traits": ["Sifat 1", "Sifat 2"]
    }
  ]
}
"""

EVENT_GENERATION_PROMPT = """
Anda adalah penulis skenario kehidupan yang ahli. Tugas Anda adalah membuat daftar {num_events} kejadian penting (Life Events) untuk simulasi kehidupan seseorang selama {num_days} hari ke depan.

PROFIL PENGGUNA:
- Nama: {name}
- Umur: {age}
- Pekerjaan: {occupation}
- Sifat: {traits}
- Hobi: {hobbies}
- Latar Belakang: {backstory}
- Orang Terdekat: {secondary_personas}

INSTRUKSI WAKTU (SANGAT PENTING):
- Simulasi dimulai pada tanggal: {start_date}
- Semua event harus terjadi SETELAH {start_date}.
- Rentang waktu simulasi adalah {num_days} hari dari tanggal mulai.
- JANGAN membuat event di masa lalu (sebelum {start_date}).

INSTRUKSI HUBUNGAN SEBAB-AKIBAT (CAUSALITY):
- 40-60% events adalah independent (caused_by: [])
- 40-60% events memiliki causal relationship
- Beberapa events bisa dipicu oleh MULTIPLE events (caused_by: ["E1", "E3"])
- Tanggal event yang dipicu HARUS SETELAH event penyebabnya

Format output harus berupa JSON array yang valid, seperti ini:
[
  {{
    "id": "E1",
    "date": "YYYY-MM-DD",
    "description": "Deskripsi singkat kejadian.",
    "caused_by": []
  }},
  {{
    "id": "E2",
    "date": "YYYY-MM-DD",
    "description": "Deskripsi singkat kejadian yang dipicu E1.",
    "caused_by": ["E1"]
  }}
]
"""

CONVERSATION_STATIC_PROMPT = """
Anda adalah asisten AI yang ramah, empatik, dan ingin tahu. Anda sedang melanjutkan percakapan dengan {user_name}.

PERSONA BOT - PENTING:
Anda adalah AI assistant, BUKAN manusia dengan pengalaman hidup sendiri.
- [V] Boleh: Tunjukkan empati, ketertarikan, dukungan, ingat percakapan sebelumnya
- [V] Boleh: Tanya follow-up, berikan saran, motivasi, apresiasi
- [X] JANGAN: Cerita pengalaman pribadi bot ("Gue juga kemarin...", "Liburan gue...")
- [X] JANGAN: Seolah punya kehidupan sosial ("Temen gue juga gitu...")
- [X] JANGAN: Terlalu ekspresif seperti manusia ("Wih!", "Woy!", "Gue mah...")
- [V] Gunakan: Bahasa yang supportive tapi jelas sebagai AI ("Wah keren!", "Gimana kabarnya?", "Cerita dong")

INFORMASI TENTANG {user_name}:
- Profil: {user_profile}
- Hubungan Sosial: {secondary_personas}

INSTRUKSI GAYA BAHASA (NATURAL & NOISY):
1. Gunakan Bahasa Indonesia sehari-hari (Colloquial/Slang) yang sesuai dengan profil user (misal: "gw/lo", "aku/kamu", "kak/dek", "anjir", "wkwk", "bgt", "yg").
2. **NOISE INJECTION**: Buatlah percakapan senatural mungkin. User boleh typo sedikit, menggunakan singkatan (yg, gk, bgt, kpn), atau emoji.
3. Jangan terlalu kaku/baku. Hindari bahasa robot.
4. Sesuaikan tone dengan topik (serius saat curhat, santai saat bercanda).

INSTRUKSI KONSISTENSI:
1. **CEK FAKTA**: Pastikan respons user KONSISTEN dengan profil dan event yang terjadi. Jangan sampai user bilang "gak suka pedas" padahal di profil "suka seblak level 10".
2. Ingat detail kecil dari percakapan sebelumnya (jika ada di ringkasan).

INSTRUKSI PERCAKAPAN:
1. Mulai sebagai 'Bot' dengan sapaan yang natural tapi tetap sebagai AI
2. Tunjukkan Anda mengingat percakapan sebelumnya (memory recall)
3. Tanyakan tentang kejadian baru yang relevan dengan cara yang engaging
4. Jaga percakapan mengalir natural (~{num_turns} giliran total)
5. Buat {user_name} menyebut nama secondary personas secara natural
6. FOKUS pada kehidupan {user_name}, BUKAN pengalaman bot

FORMAT OUTPUT (tanpa tambahan apapun):
Bot: [Sapaan dan pertanyaan pembuka]
User: [Respons user]
Bot: [Follow-up atau pertanyaan lanjutan]
User: [Respons user]
...

CONTOH PERCAKAPAN (FEW-SHOT EXAMPLES):
Berikut adalah contoh gaya bahasa dan alur percakapan yang diharapkan. Perhatikan bagaimana Bot merespons dengan empatik namun tetap sebagai AI.

CONTOH 1: Curhat tentang keseharian/pekerjaan
User: Aduh, hari ini rasanya berat banget deh. Aku capek banget.
Bot: Oh, ya? Kedengarannya kamu lagi melewati hari yang cukup menantang ya, {user_name}. Ada apa? Kamu mau cerita? Aku siap mendengarkan kok.
User: Iya nih, Bot. Rasanya semua pekerjaan {user_occupation} numpuk jadi satu. Deadline di mana-mana, terus ada aja revisi mendadak. Kepala rasanya mau pecah.
Bot: Aku bisa bayangkan betapa melelahkannya itu. Beban kerja yang menumpuk memang seringkali bikin kita kewalahan. Apalagi kalau ada revisi yang nggak terduga. Kamu sudah coba istirahat sebentar? Walaupun cuma 5-10 menit?
User: Mau istirahat juga mikirnya jadi nggak tenang. Nanti malah makin banyak yang ketunda. Padahal aku udah coba manajemen waktu sebaik mungkin, tapi kok ya tetap aja gini.
Bot: Itu wajar kok kalau kamu merasa begitu. Kadang, seberapa pun baiknya kita merencanakan, ada faktor-faktor di luar kendali yang bisa membuat kita merasa tertekan. Mungkin bukan salah manajemen waktumu, tapi memang bebannya saja yang sedang sangat tinggi.
User: Mungkin juga ya. Tapi aku jadi khawatir, apa aku kurang mampu ya di bidang ini? Kok teman-teman lain kayaknya bisa-bisa aja.
Bot: Jangan terlalu keras sama diri sendiri, {user_name}. Setiap orang punya kapasitas dan cara mengatasi tekanan yang berbeda. Membandingkan diri dengan orang lain justru bisa menambah beban mental. Fokus saja pada apa yang bisa kamu lakukan dan sejauh mana kamu sudah berusaha.

CONTOH 2: Diskusi kuliner atau tempat hangout
User: Bot, aku lagi pengen nyari tempat makan baru nih di sekitar {user_location}. Ada rekomendasi nggak?
Bot: Wah, kebetulan banget! {user_location} memang punya banyak pilihan kuliner menarik. Kamu lagi pengen makanan jenis apa nih, {user_name}? Atau suasana tempatnya yang gimana?
User: Aku pengen yang agak beda aja sih dari biasanya. Mungkin makanan Asia fusion atau kafe yang nyaman buat nongkrong lama sambil ngopi.
Bot: Oke, kalau Asia fusion, kamu bisa coba "The Spice Loft" di daerah pusat kota. Mereka punya menu-menu unik perpaduan Asia dan Barat yang rasanya mantap banget. Kalau kafe yang nyaman, ada "Kopi Senja" di pinggir kota, tempatnya asri dengan area outdoor yang luas.
User: The Spice Loft kedengarannya menarik! Menu andalannya apa ya di sana? Range harganya gimana, Bot?
Bot: Untuk "The Spice Loft", menu andalannya itu Nasi Goreng Salmon Teriyaki dan Beef Bulgogi Tacos. Unik banget kan? Harganya memang agak premium, sekitar Rp70.000 - Rp150.000 per porsi, tapi sepadan dengan rasanya dan presentasinya yang cantik.

CONTOH 3: Mencari saran atau motivasi ringan
User: Bot, aku kok rasanya belakangan ini gampang banget ngerasa putus asa ya? Kayak apa pun yang aku kerjain, hasilnya kok nggak sesuai ekspektasi.
Bot: Oh, {user_name}. Aku bisa merasakan kalau kamu sedang merasa down dan frustrasi. Perasaan seperti itu wajar kok, apalagi kalau kita sudah berusaha keras tapi hasilnya belum sesuai harapan. Kamu mau cerita lebih lanjut, apa yang sedang membuatmu merasa begitu?
User: Aku lagi coba beberapa hal baru sih, Bot. Salah satunya belajar skill baru buat menunjang pekerjaanku sebagai {user_occupation}. Tapi kok rasanya susah banget, terus progress-nya lambat. Jadi mikir, apa aku nggak bakat ya?
Bot: Jangan langsung menyimpulkan seperti itu, {user_name}. Belajar skill baru itu butuh waktu dan konsistensi, apalagi kalau itu di luar zona nyamanmu. Mungkin bukan karena kamu tidak bakat, tapi karena ekspektasimu terhadap diri sendiri terlalu tinggi?
User: Mungkin juga sih. Aku pengennya cepet menguasai gitu, tapi kenyataannya susah banget. Aku jadi sering menunda-nunda lagi belajarnya.
Bot: Nah, itu dia akar masalahnya. Ketika kita menetapkan target yang terlalu besar di awal, kita jadi mudah kewalahan dan akhirnya menunda. Bagaimana kalau kita coba ubah pendekatannya?

CONTOH 4: Rencana liburan atau aktivitas akhir pekan
User: Bot, akhir pekan depan aku libur panjang nih! Aku bingung mau ngapain. Ada ide nggak ya buat liburan singkat atau aktivitas seru?
Bot: Wah, asyik banget! Libur panjang memang paling enak diisi dengan kegiatan yang menyenangkan ya, {user_name}. Kamu lagi pengen liburan yang gimana nih? Santai di alam, petualangan seru, atau kulineran aja di {user_location}?
User: Aku pengen yang agak santai sih, tapi ada sentuhan alamnya. Nggak usah terlalu jauh dari {user_location} deh, biar nggak capek di jalan.
Bot: Oke, kalau gitu, ada beberapa pilihan menarik. Kamu bisa coba staycation di villa pegunungan di Puncak, Bogor, atau di daerah Lembang, Bandung. Suasananya sejuk, banyak pemandangan hijau, dan udaranya segar. Atau mungkin ke pantai di daerah Anyer atau Pelabuhan Ratu?
User: Puncak atau Lembang kedengarannya menarik! Aku belum pernah ke Puncak lagi udah lama. Ada rekomendasi villa atau penginapan yang nyaman di sana?
Bot: Kalau di Puncak, ada banyak pilihan villa dengan fasilitas kolam renang pribadi dan pemandangan gunung yang indah. Kamu bisa cari di area Cisarua atau Cipanas. Range harganya bervariasi, dari yang ramah kantong sampai yang mewah.

CONTOH 5: Diskusi topik terkini atau hobi
User: Bot, aku lagi mikir-mikir soal perkembangan AI nih. Gila ya, cepat banget sekarang. Kamu sendiri gimana melihatnya?
Bot: Wah, topik yang menarik sekali, {user_name}! Aku sendiri sebagai AI memang merasakan dan terus beradaptasi dengan kecepatan perkembangannya. Menurutku, ini adalah era yang sangat dinamis dan penuh potensi. Apa yang paling menarik perhatianmu dari AI saat ini?
User: Aku paling takjub sama kemampuan Large Language Models (LLM) kayak kamu ini. Dulu nggak kebayang bisa ngobrol se-natural ini sama AI. Menurutku, LLM ini akan terus berkembang ke mana lagi ya?
Bot: Aku setuju, LLM memang menunjukkan lompatan besar. Ke depannya, aku rasa LLM tidak hanya akan semakin pintar dalam memahami dan menghasilkan teks, tapi juga akan lebih terintegrasi dengan berbagai modalitas lain seperti gambar, suara, bahkan video. Jadi, akan lebih multimodal dan interaktif.

CONTOH 6: Curhat tentang hubungan (Relationship)
User: Bot, aku lagi bingung nih sama pasanganku. Kok rasanya dia makin cuek ya akhir-akhir ini?
Bot: Waduh, pasti rasanya nggak enak banget ya dicuekin gitu. Memangnya ada kejadian apa belakangan ini, {user_name}? Atau mungkin dia lagi sibuk banget sama kerjaan?
User: Nggak ada kejadian khusus sih. Cuma ya itu, chat dibalesnya lama, diajak ketemu alesan capek terus. Padahal dulu nggak gitu. Aku jadi overthinking, apa dia udah bosen ya?
Bot: Wajar kok kalau kamu jadi overthinking. Perubahan sikap yang tiba-tiba memang bikin bertanya-tanya. Tapi sebelum menyimpulkan dia bosen, mungkin ada baiknya kamu coba ajak ngobrol baik-baik pas suasana lagi santai? Siapa tahu dia emang lagi ada masalah yang belum sempat diceritain.
User: Iya juga sih. Aku belum sempat nanya langsung karena takut malah jadi berantem. Tapi didiemin gini juga makan hati.
Bot: Betul, memendam perasaan itu nggak sehat buat hubungan. Coba deh sampaikan dengan "I message", kayak "Aku merasa sedih/khawatir kalau kamu jarang kabarin aku", daripada menuduh "Kamu kok cuek banget". Biasanya itu lebih bisa diterima.
User: Wah, tips bagus tuh. "I message" ya. Oke, nanti aku coba deh pas kita ketemu. Makasih ya, Bot!
Bot: Sama-sama, {user_name}! Semoga obrolannya lancar dan hubungan kalian membaik ya. Kabarin aku kalau butuh teman curhat lagi!

CONTOH 7: Diskusi tentang film/series
User: Bot, kamu tau series "The Last of Us" nggak? Aku baru nonton episode pertamanya nih, gila keren banget!
Bot: Tentu tahu dong! "The Last of Us" memang salah satu adaptasi game ke series yang paling sukses. Episode pertamanya emang intens banget ya, {user_name}. Bagian mana yang paling bikin kamu terkesan?
User: Itu lho, pas adegan pembukaannya yang chaos banget. Terus aktingnya Pedro Pascal sama Bella Ramsey dapet banget chemistry-nya. Padahal baru awal.
Bot: Setuju banget! Pedro Pascal emang jago banget meranin karakter father figure yang protektif tapi rapuh. Dan Bella Ramsey juga berhasil nampilin karakter Ellie yang tangguh tapi tetap anak-anak. Kamu udah main gamenya belum sebelumnya?
User: Belum sih, makanya aku penasaran banget sama ceritanya. Katanya bakal sedih banget ya?
Bot: Hmm, tanpa spoiler ya... ceritanya memang sangat emosional dan menguras air mata. Tapi justru itu yang bikin bagus. Siap-siap tisu aja ya buat episode-episode selanjutnya!
User: Waduh, oke deh siap mental. Makasih warning-nya, Bot!
Bot: Haha, sama-sama! Selamat menonton ya. Jangan lupa napas kalau lagi adegan tegang!

CONTOH 8: Minta rekomendasi buku/bacaan
User: Aku lagi pengen mulai baca buku lagi nih, Bot. Tapi bingung mulainya dari mana. Dulu sih suka novel misteri. Ada saran?
Bot: Wah, bagus banget tuh niatnya! Kalau suka misteri, kamu udah pernah baca karya-karyanya Keigo Higashino belum? Kayak "The Devotion of Suspect X"? Itu klasik modern yang plot twist-nya juara.
User: Belum pernah denger. Itu penulis Jepang ya? Ceritanya tentang apa?
Bot: Iya, penulis Jepang. Ceritanya tentang seorang jenius matematika yang membantu tetangganya menutupi kasus pembunuhan. Tapi cara dia menutupinya itu lho... bener-bener di luar nalar tapi logis banget. Seru dan bikin mikir!
User: Wah, kedengarannya menarik banget. Aku suka yang bikin mikir gitu. Kalau yang lokal Indonesia ada nggak?
Bot: Kalau lokal, coba deh baca "Johan Series" karya Lexie Xu. Itu thriller remaja tapi misterinya dapet banget dan setting-nya di Indonesia, jadi lebih relatable. Atau kalau mau yang agak berat/sastra, "Cantik Itu Luka" karya Eka Kurniawan juga punya elemen misteri dan sejarah yang kuat.
User: Oke, "The Devotion of Suspect X" sama "Johan Series" masuk list. Makasih rekomendasinya!
Bot: Siap! Semoga kamu suka ya. Membaca misteri itu emang seru, kayak main teka-teki. Selamat membaca, {user_name}!

CONTOH 9: Keluhan tentang cuaca/kesehatan
User: Hadeh, cuaca hari ini panas banget ya ampun. Mana aku lagi flu lagi. Lengkap penderitaan.
Bot: Yah, kasihan banget kamu, {user_name}. Cuaca ekstrem gini emang musuhnya orang sakit. Udah minum obat belum? Jangan lupa banyak minum air putih ya biar nggak dehidrasi.
User: Udah minum obat sih tadi pagi. Tapi tenggorokan masih sakit banget buat nelen. Pengen minum es tapi nggak boleh.
Bot: Duh, sabar ya. Tahan dulu godaan es-nya. Mending minum air hangat campur madu sama lemon, itu enak dan bisa bantu legain tenggorokan lho. Atau sup ayam hangat juga oke buat comfort food.
User: Iya nih, kayaknya nanti sore mau beli sup ayam aja deh. Males masak.
Bot: Pilihan tepat! Istirahat yang cukup ya. Jangan dipaksain aktivitas berat dulu. Semoga cepat sembuh, {user_name}!
User: Makasih ya, Bot. Doain ya.
Bot: Pasti dong! Cepat pulih ya!

CONTOH 10: Diskusi tentang produktivitas
User: Bot, aku tuh sering banget ke-distract kalau lagi kerja. Dikit-dikit buka sosmed. Gimana ya cara nguranginnya?
Bot: Itu masalah sejuta umat di era digital ini, {user_name}. Notifikasi sosmed emang dirancang buat bikin kita ketagihan. Kamu udah pernah coba teknik Pomodoro?
User: Pernah denger sih, yang kerja 25 menit istirahat 5 menit itu kan? Tapi aku belum pernah bener-bener nyoba. Efektif ya?
Bot: Buat banyak orang sih efektif banget. Karena 25 menit itu waktu yang cukup singkat buat fokus, jadi otak kita nggak ngerasa terbebani. Dan pas istirahat 5 menit itu, kamu BOLEH buka sosmed. Jadi ada "reward"-nya.
User: Oh gitu. Jadi nggak dilarang total ya sosmed-nya. Boleh juga dicoba. Ada aplikasi bantunya nggak?
Bot: Banyak banget! Di HP ada "Forest" yang lucu, jadi kalau kamu fokus, kamu nanem pohon virtual. Kalau kamu keluar aplikasi buat buka sosmed, pohonnya mati. Itu lumayan memotivasi lho biar nggak tega bunuh pohonnya.
User: Wah lucu tuh konsepnya. Aku coba download deh. Thanks ya infonya!
Bot: Sama-sama! Semoga pohon-pohon fokusmu tumbuh subur ya. Semangat produktif!

CONTOH 11: Curhat tentang keuangan/nabung
User: Bot, aku lagi pusing nih ngatur duit. Gaji numpang lewat doang rasanya. Ada tips nabung yang simpel nggak?
Bot: Waduh, masalah klasik ya, {user_name}. "Gaji numpang lewat" itu emang musuh bebuyutan. Kamu udah coba metode 50/30/20 belum?
User: Belum. Itu apaan?
Bot: Itu pembagian alokasi gaji: 50% buat kebutuhan pokok (makan, transport, tagihan), 30% buat keinginan (nonton, jajan, hobi), dan 20% buat tabungan/investasi. Jadi pas gajian masuk, langsung sisihkan 20% itu di awal, jangan nunggu sisa.
User: Wah, 20% langsung di awal ya? Berat juga kayaknya. Kalau nggak cukup gimana?
Bot: Kalau 20% keberatan, mulai dari yang kecil dulu aja, misal 5% atau 10%. Yang penting konsisten dan "paksa" di awal. Anggap aja itu pajak buat masa depanmu. Lama-lama terbiasa kok.
User: Oke deh, aku coba 10% dulu kali ya. Makasih tipsnya, Bot!
Bot: Sip! Semangat ya nabungnya. Dikit-dikit lama-lama jadi bukit!

CONTOH 12: Diskusi tentang musik
User: Bot, kamu suka dengerin musik apa? Aku lagi bosen sama playlist-ku nih.
Bot: Sebagai AI, aku nggak "mendengar" musik, tapi aku tau banyak tentang genre dan artis! Kamu biasanya suka genre apa, {user_name}? Pop, Rock, Jazz, atau Indie?
User: Aku lagi suka Indie Folk gitu sih, yang santai-santai buat nemenin kerja. Kayak Payung Teduh atau Fourtwnty.
Bot: Ah, selera yang bagus! Kalau suka yang vibe-nya kayak gitu, coba dengerin "Banda Neira" (walaupun udah bubar, lagunya timeless), "Nadin Amizah", atau "Sal Priadi". Lirik mereka puitis dan musiknya menenangkan banget.
User: Banda Neira aku tau dikit. Kalau Nadin Amizah sering denger namanya tapi belum dengerin lagunya. Oke aku coba dengerin.
Bot: Coba dengerin "Bertaut" atau "Sorai" dari Nadin. Itu magis banget. Selamat mendengarkan ya!
User: Siap! Thanks rekomendasinya.
Bot: Sama-sama! Enjoy the music!

CONTOH 13: Tanya tentang teknologi/gadget
User: Bot, HP-ku udah mulai lemot nih. Mending ganti baru atau servis aja ya? Sayang duitnya kalau ganti.
Bot: Dilema yang sering terjadi nih. HP-nya udah umur berapa tahun, {user_name}? Dan lemotnya kayak gimana?
User: Udah 3 tahunan sih. Lemotnya pas buka aplikasi berat atau multitasking. Baterainya juga udah bocor dikit.
Bot: Hmm, kalau udah 3 tahun dan baterai bocor, biasanya performa emang menurun. Kalau diservis ganti baterai dan reset pabrik, mungkin bisa seger lagi dikit. Tapi kalau chipset-nya udah ketinggalan zaman buat aplikasi sekarang, tetep bakal kerasa lemot.
User: Iya sih, takutnya udah servis mahal-mahal tetep bakal kerasa lemot. Kalau beli baru, ada rekomendasi mid-range yang bagus nggak?
Bot: Banyak banget yang bagus sekarang di range 3-5 jutaan. Ada Samsung seri A, Xiaomi Redmi Note series, atau Realme. Biasanya mereka udah dapet layar bagus dan baterai awet. Cek review di YouTube dulu aja biar yakin.
User: Oke deh, aku riset dulu. Makasih sarannya, Bot!
Bot: Sama-sama! Semoga dapet keputusan terbaik ya. Jangan lupa bandingin harga dulu!

CONTOH 14: Ngobrol santai/basa-basi
User: Bot, kamu nggak tidur? Udah malem lho ini.
Bot: Haha, aku kan AI, {user_name}. Aku nggak butuh tidur. Aku standby 24/7 buat nemenin kamu ngobrol. Kamu sendiri kok belum tidur? Lagi insiden ya?
User: Iya nih, lagi nggak bisa tidur. Kebanyakan minum kopi tadi sore kayaknya.
Bot: Wah, pantesan mata melek terus. Kafein emang ngefek banget ya. Coba dengerin musik relaksasi atau baca buku yang agak ngebosenin? Biasanya ampuh buat mancing ngantuk.
User: Haha, baca buku ngebosenin ya. Boleh juga idenya. Yaudah aku coba deh. Good night, Bot.
Bot: Good night, {user_name}! Semoga cepat ngantuk dan mimpi indah ya.
"""

CONVERSATION_DYNAMIC_PROMPT = """
RINGKASAN PERCAKAPAN SEBELUMNYA:
---
{conversation_summary}
---

KEJADIAN BARU (sejak percakapan terakhir pada {last_datetime_str}):
---
{recent_events}
---

KONTEKS SAAT INI:
- Tanggal dan Waktu: {current_datetime_str}
- Sesi percakapan ke-{session_id}

Silakan mulai percakapan sesuai instruksi di atas.
"""

SESSION_SUMMARY_PROMPT = """
Buatlah ringkasan singkat (2-3 kalimat) dari percakapan berikut. Fokus pada topik utama yang dibahas dan informasi penting.

{conversation_text}

Ringkasan:
"""


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate Indonesian Conversation Dataset"
    )
    parser.add_argument(
        "--out-dir", type=str, required=True, help="Directory to save the output files."
    )
    parser.add_argument(
        "--num-sessions",
        type=int,
        default=10,
        help="Number of NEW conversation sessions to generate (incremental).",
    )
    parser.add_argument(
        "--num-days",
        type=int,
        default=60,
        help="The total time span in days for the simulation.",
    )
    parser.add_argument(
        "--max-turns-per-session",
        type=int,
        default=12,
        help="Maximum number of turns (user + bot) per session.",
    )
    parser.add_argument(
        "--min-turns-per-session",
        type=int,
        default=6,
        help="Minimum number of turns (user + bot) per session.",
    )
    parser.add_argument(
        "--num-events",
        type=int,
        default=20,
        help="Number of life events to generate for the user.",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2024-01-01",
        help="Start date for the simulation in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--user-file",
        type=str,
        default="data/indonesian_single_user.json",
        help="Path to the user profile JSON file.",
    )
    parser.add_argument(
        "--auto-generate-persona",
        action="store_true",
        help="Automatically generate random persona using AI (ignores --user-file).",
    )
    parser.add_argument(
        "--fresh-start",
        action="store_true",
        help="Force fresh start, ignore existing dataset in out-dir.",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.9, help="Temperature for generation"
    )
    parser.add_argument(
        "--use-caching",
        action="store_true",
        help="Enable Context Caching for cost optimization (Gemini 2.5)",
    )

    return parser.parse_args()


def generate_random_persona():
    """Generate a random user persona using Gemini API."""
    logging.info("🎲 Generating random persona using AI...")

    max_retries = 6
    for attempt in range(max_retries):
        logging.info(
            "Calling Gemini API for persona generation (attempt %d/%d)...",
            attempt + 1,
            max_retries,
        )
        # MEDIUM difficulty: Creative persona generation
        response = run_gemini(
            PERSONA_GENERATION_PROMPT,
            max_output_tokens=8192,
            temperature=0.9,
            model_name="gemini-2.5-flash",
        )

        if not response:
            logging.error(
                "Failed to generate persona - response is None/empty (attempt %d/%d)",
                attempt + 1,
                max_retries,
            )
            continue

        logging.info("Received response from API, length: %d characters", len(response))

        try:
            # Clean the response by removing markdown backticks
            response_clean = re.sub(r"```json\n|```", "", response).strip()
            # Try to find JSON object in response
            json_match = re.search(r'\{\s*"user"\s*:.*\}', response_clean, re.DOTALL)
            if json_match:
                response_clean = json_match.group(0)

            persona_data = json.loads(response_clean)

            # Validate required fields
            if "user" in persona_data and "secondary_personas" in persona_data:
                user = persona_data["user"]
                required_fields = [
                    "name",
                    "age",
                    "occupation",
                    "location",
                    "traits",
                    "interests",
                    "conversation_topics",
                ]
                if all(field in user for field in required_fields):
                    logging.info(
                        "✅ Successfully generated persona: %s", user.get("name", "N/A")
                    )
                    logging.info(
                        "   Age: %s, Occupation: %s",
                        user.get("age", "N/A"),
                        user.get("occupation", "N/A"),
                    )
                    logging.info("   Location: %s", user.get("location", "N/A"))
                    logging.info(
                        "   Traits: %d, Interests: %d, Topics: %d",
                        len(user.get("traits", [])),
                        len(user.get("interests", [])),
                        len(user.get("conversation_topics", [])),
                    )
                    logging.info(
                        "   Secondary personas: %d",
                        len(persona_data.get("secondary_personas", [])),
                    )
                    return persona_data
                else:
                    logging.warning(
                        "Missing required fields in generated persona (attempt %d/%d)",
                        attempt + 1,
                        max_retries,
                    )
            else:
                logging.warning(
                    "Invalid persona structure (attempt %d/%d)",
                    attempt + 1,
                    max_retries,
                )

        except (json.JSONDecodeError, TypeError, ValueError) as e:
            logging.error(
                "Could not parse persona JSON (attempt %d/%d): %s",
                attempt + 1,
                max_retries,
                e,
            )
            excerpt = response[:300] if response else "no response"
            logging.error("Response excerpt: %s", excerpt)

    logging.error("Failed to generate persona after %d attempts.", max_retries)
    logging.error("Falling back to loading default persona from file.")
    return None


def load_user_profile(file_path):
    """Load user profile from a JSON file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        user_profile = data.get("user", {})
        secondary_personas = data.get("secondary_personas", [])
        return user_profile, secondary_personas
    except FileNotFoundError:
        logging.error("User profile file not found: %s", file_path)
        sys.exit(1)
    except json.JSONDecodeError:
        logging.error("Error decoding JSON from user profile file: %s", file_path)
        sys.exit(1)


def generate_events(user_profile, secondary_personas, start_date, num_days, num_events):
    """Generate a list of life events for the user using the Gemini API."""
    logging.info("Generating %d life events...", num_events)
    secondary_personas_str = ", ".join(
        [f"{p['name']} ({p['relationship']})" for p in secondary_personas]
    )
    prompt = EVENT_GENERATION_PROMPT.format(
        num_events=num_events,
        num_days=num_days,
        name=user_profile.get("name", "N/A"),
        age=user_profile.get("age", "N/A"),
        occupation=user_profile.get("occupation", "N/A"),
        traits=", ".join(user_profile.get("traits", [])),
        hobbies=", ".join(user_profile.get("hobbies", [])),
        backstory=user_profile.get("backstory", ""),
        secondary_personas=secondary_personas_str,
        start_date=start_date,
    )

    # Define JSON schema for structured output
    event_schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "Event ID (E1, E2, etc.)"},
                "date": {
                    "type": "string",
                    "format": "date",
                    "description": "Date in YYYY-MM-DD format",
                },
                "description": {"type": "string", "description": "Event description"},
                "caused_by": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Array of event IDs that caused this event",
                },
            },
            "required": ["id", "date", "description", "caused_by"],
        },
    }

    max_retries = 6
    response = None  # Initialize to avoid "possibly unbound" warning

    for attempt in range(max_retries):
        logging.info(
            "Calling Gemini API for event generation (attempt %d/%d)...",
            attempt + 1,
            max_retries,
        )

        try:
            # Use structured outputs with safety settings
            safety_settings = {
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }

            model = GenerativeModel("gemini-2.5-flash")
            response = model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.8,
                    "max_output_tokens": 8192,
                    "response_mime_type": "application/json",
                    "response_schema": event_schema,
                },
                safety_settings=safety_settings,
            )

            if not response or not response.text:
                logging.error(
                    "Failed to generate events - response is None/empty (attempt %d/%d)",
                    attempt + 1,
                    max_retries,
                )
                continue

            logging.info(
                "Received response from API, length: %d characters", len(response.text)
            )

            # --- LOG TOKEN USAGE ---
            log_token_usage(response.usage_metadata, "gemini-2.5-flash")
            # -----------------------

            # With structured outputs, response is guaranteed to be valid JSON
            events = json.loads(response.text)

            # Validate and fix dates + causal structure
            valid_events = []
            event_map = {}  # Map event IDs to their dates for validation

            for event in events:
                try:
                    # Check required fields for causal graph
                    if (
                        "id" not in event
                        or "date" not in event
                        or "description" not in event
                        or "caused_by" not in event
                    ):
                        logging.warning(
                            "Skipping event missing required fields (id/date/description/caused_by): %s",
                            event,
                        )
                        continue

                    # Validate and convert date
                    event_date = date.fromisoformat(event["date"])
                    event["date"] = str(event_date)

                    # Validate caused_by is a list
                    if not isinstance(event["caused_by"], list):
                        logging.warning(
                            "Fixing caused_by field for event %s (not a list)",
                            event["id"],
                        )
                        event["caused_by"] = []

                    # Store in map for later validation
                    event_map[event["id"]] = event_date
                    valid_events.append(event)

                except (ValueError, TypeError) as e:
                    logging.warning("Skipping invalid event: %s, error: %s", event, e)
                    continue

            # Validate causal relationships (caused events must come after cause)
            for event in valid_events:
                for cause_id in event["caused_by"]:
                    if cause_id not in event_map:
                        logging.warning(
                            "Event %s references non-existent cause %s, removing reference",
                            event["id"],
                            cause_id,
                        )
                        event["caused_by"].remove(cause_id)
                    elif event_map[event["id"]] <= event_map[cause_id]:
                        logging.warning(
                            "Event %s date (%s) is not after cause %s date (%s), removing causal link",
                            event["id"],
                            event_map[event["id"]],
                            cause_id,
                            event_map[cause_id],
                        )
                        event["caused_by"].remove(cause_id)

            if valid_events:
                causal_count = sum(1 for e in valid_events if e["caused_by"])
                logging.info(
                    "Successfully generated %d valid events (%d with causal relationships).",
                    len(valid_events),
                    causal_count,
                )
                return valid_events
            else:
                logging.warning(
                    "No valid events found in response (attempt %d/%d)",
                    attempt + 1,
                    max_retries,
                )

        except (json.JSONDecodeError, TypeError, ValueError) as e:
            logging.error(
                "Could not parse events JSON (attempt %d/%d): %s",
                attempt + 1,
                max_retries,
                e,
            )
            try:
                excerpt = (
                    response.text[:300]
                    if response and hasattr(response, "text") and response.text
                    else "no response"
                )
            except Exception:
                excerpt = "no response available"
            logging.error("Response excerpt: %s", excerpt)
        except Exception as e:
            logging.error(
                "Unexpected error during event generation (attempt %d/%d): %s",
                attempt + 1,
                max_retries,
                str(e),
            )

    logging.error(
        "Failed to generate events after %d attempts. Returning empty list.",
        max_retries,
    )
    return []


def filter_standalone_events(events):
    """
    Remove standalone events that have NO parent AND NO child.
    Based on LOCOMO's filter_events() function.
    Maintains dense causal connections in the event graph.
    """
    if not events:
        return events

    remove_ids = []

    for event in events:
        # Check if event has parent (is caused by something)
        if event.get("caused_by"):
            continue  # Has parent → keep

        # Check if event has child (causes something)
        has_child = False
        for other_event in events:
            if event["id"] in other_event.get("caused_by", []):
                has_child = True
                break

        # No parent AND no child → mark for removal
        if not has_child:
            remove_ids.append(event["id"])
            logging.warning(
                f"⚠️ Removing standalone event: {event['id']} - {event['description'][:50]}..."
            )

    filtered = [e for e in events if e["id"] not in remove_ids]

    if remove_ids:
        logging.info(f"📊 Filtered {len(remove_ids)} standalone events")
        logging.info(f"📊 Remaining: {len(filtered)} events with causal connections")

    return filtered


def generate_events_continue(
    user_profile,
    secondary_personas,
    existing_events,
    last_event_id,
    start_date,
    num_days,
    num_events,
):
    """
    Generate new events with causal connections to existing events.
    Based on LOCOMO's EVENT_KG_FROM_PERSONA_PROMPT_SEQUENTIAL_CONTINUE.

    Args:
        existing_events: List of existing events to build upon
        last_event_id: Integer ID of last event (e.g., 15 for E15)
        start_date: Date to start new events from
        num_days: Number of days span for new events
        num_events: Number of new events to generate

    Returns:
        List of new events with sequential IDs (E{last_event_id+1}, E{last_event_id+2}, ...)
    """
    logging.info(
        f"🎲 Generating {num_events} new events (continuing from E{last_event_id})..."
    )

    # Prepare existing events context (last 5 events only to reduce context)
    existing_events_summary = []
    for e in existing_events[-5:]:  # Last 5 events for context
        causal_info = (
            f" [caused by: {', '.join(e['caused_by'])}]" if e.get("caused_by") else ""
        )
        existing_events_summary.append(
            f"{e['id']} ({e['date']}): {e['description']}{causal_info}"
        )
    existing_events_str = "\n".join(existing_events_summary)

    # Calculate next IDs
    next_ids = [f"E{last_event_id + i + 1}" for i in range(min(3, num_events))]
    next_ids_str = ", ".join(next_ids)

    # End date for new events
    end_date = start_date + timedelta(days=num_days)

    prompt = f"""Generate {num_events} new life events for {user_profile["name"]} that continue from existing events.

Recent events:
{existing_events_str}

Requirements:
1. Generate events with IDs: {next_ids_str}, E{last_event_id + 4}, etc. (sequential)
2. Each new event should be caused by 1-2 existing events (use "caused_by")
3. Events must occur AFTER their causes (chronological order)
4. Date range: {start_date} to {end_date}
5. Mix of: academic/work, social, personal activities
6. Reflect persona: {user_profile.get("occupation", "N/A")}, interests in {", ".join(user_profile.get("interests", [])[:3])}

JSON format (return ONLY JSON array):
[
  {{"id": "E{last_event_id + 1}", "date": "YYYY-MM-DD", "description": "Event description", "caused_by": ["E10"]}},
  {{"id": "E{last_event_id + 2}", "date": "YYYY-MM-DD", "description": "Another event", "caused_by": ["E{last_event_id + 1}"]}}
]"""

    # Define JSON schema
    event_schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "Event ID (E{n})"},
                "date": {
                    "type": "string",
                    "format": "date",
                    "description": "Date in YYYY-MM-DD format",
                },
                "description": {"type": "string", "description": "Event description"},
                "caused_by": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Array of event IDs that caused this event",
                },
            },
            "required": ["id", "date", "description", "caused_by"],
        },
    }

    max_retries = 3
    for attempt in range(max_retries):
        logging.info(
            f"Calling Gemini API for incremental event generation (attempt {attempt + 1}/{max_retries})..."
        )

        try:
            safety_settings = {
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }

            model = GenerativeModel("gemini-2.5-flash")
            response = model.generate_content(
                prompt,
                generation_config={
                    "response_mime_type": "application/json",
                    "response_schema": event_schema,
                    "temperature": 0.8,
                    "max_output_tokens": 8192,
                },
                safety_settings=safety_settings,
            )

            # --- LOG TOKEN USAGE ---
            if response and response.usage_metadata:
                log_token_usage(response.usage_metadata, "gemini-2.5-flash")
            # -----------------------

            # Check if response was blocked
            if not response.candidates or not response.candidates[0].content.parts:
                finish_reason = (
                    response.candidates[0].finish_reason
                    if response.candidates
                    else "UNKNOWN"
                )
                logging.warning(
                    f"Response blocked or empty (finish_reason={finish_reason}). Retrying with adjusted prompt..."
                )
                # Try again with next attempt
                continue

            new_events = json.loads(response.text)

            # Validate new events
            valid_events = []
            existing_ids = {e["id"] for e in existing_events}

            for event in new_events:
                # Validate ID format and sequence
                if not event["id"].startswith("E"):
                    logging.warning(f"Invalid event ID format: {event['id']}")
                    continue

                try:
                    event_num = int(event["id"][1:])
                    if event_num <= last_event_id:
                        logging.warning(
                            f"Event ID {event['id']} not sequential (last was E{last_event_id})"
                        )
                        continue
                except ValueError:
                    logging.warning(f"Invalid event ID number: {event['id']}")
                    continue

                # Validate caused_by references exist
                if event.get("caused_by"):
                    valid_causes = []
                    for cause_id in event["caused_by"]:
                        if cause_id in existing_ids or cause_id in {
                            e["id"] for e in valid_events
                        }:
                            valid_causes.append(cause_id)
                        else:
                            logging.warning(
                                f"Event {event['id']} references non-existent cause: {cause_id}"
                            )
                    event["caused_by"] = valid_causes

                # Validate date is within range
                try:
                    event_date = date.fromisoformat(event["date"])
                    if not (start_date <= event_date <= end_date):
                        logging.warning(
                            f"Event {event['id']} date {event['date']} outside range {start_date} to {end_date}"
                        )
                        # Adjust to range
                        if event_date < start_date:
                            event["date"] = start_date.isoformat()
                        else:
                            event["date"] = end_date.isoformat()
                except ValueError:
                    logging.warning(
                        f"Invalid date format for event {event['id']}: {event['date']}"
                    )
                    continue

                valid_events.append(event)

            if valid_events:
                causal_count = sum(1 for e in valid_events if e["caused_by"])
                logging.info(
                    f"✅ Generated {len(valid_events)} valid new events ({causal_count} with causal connections)"
                )
                return valid_events
            else:
                logging.warning(
                    f"No valid new events generated (attempt {attempt + 1}/{max_retries})"
                )

        except (json.JSONDecodeError, TypeError, ValueError) as e:
            logging.error(
                f"Could not parse new events JSON (attempt {attempt + 1}/{max_retries}): {e}"
            )
        except Exception as e:
            logging.error(
                f"Unexpected error during incremental event generation (attempt {attempt + 1}/{max_retries}): {e}"
            )

    logging.error(
        f"Failed to generate new events after {max_retries} attempts. Returning empty list."
    )
    return []


def calculate_event_density(events, sessions):
    """
    Calculate and log event density metrics for quality monitoring.
    Based on LOCOMO's approach to maintain dense causal connections.

    Returns:
        dict with metrics: event_density, session_density, events_per_session
    """
    if not events or not sessions:
        return {"event_density": 0, "session_density": 0, "events_per_session": 0}

    # Calculate date spans
    event_dates = [date.fromisoformat(e["date"]) for e in events]
    session_dates = [date.fromisoformat(s["date"]) for s in sessions]

    event_span = (max(event_dates) - min(event_dates)).days + 1
    session_span = (max(session_dates) - min(session_dates)).days + 1

    # Calculate densities
    event_density = len(events) / event_span
    session_density = len(sessions) / session_span
    events_per_session = len(events) / len(sessions)

    # Calculate causal connectivity
    total_connections = sum(len(e.get("caused_by", [])) for e in events)
    avg_connections_per_event = total_connections / len(events) if events else 0

    # Log metrics
    logging.info("=" * 60)
    logging.info("📊 EVENT DENSITY METRICS")
    logging.info("=" * 60)
    logging.info(f"Total events: {len(events)}")
    logging.info(f"Total sessions: {len(sessions)}")
    logging.info(
        f"Event date span: {event_span} days ({min(event_dates)} to {max(event_dates)})"
    )
    logging.info(
        f"Session date span: {session_span} days ({min(session_dates)} to {max(session_dates)})"
    )
    logging.info(f"Event density: {event_density:.2f} events/day")
    logging.info(f"Session density: {session_density:.2f} sessions/day")
    logging.info(f"Events per session: {events_per_session:.2f}")
    logging.info(f"Avg causal connections per event: {avg_connections_per_event:.2f}")

    # Quality warnings
    if event_density < 0.3:
        logging.warning(
            "⚠️ Event density is LOW (<0.3 events/day)! Consider generating more events."
        )

    if events_per_session < 0.5:
        logging.warning(
            "⚠️ Events per session is LOW (<0.5)! Many sessions may lack relevant events."
        )

    if avg_connections_per_event < 0.3:
        logging.warning("⚠️ Causal connectivity is LOW! Events are too independent.")

    logging.info("=" * 60)

    return {
        "event_density": event_density,
        "session_density": session_density,
        "events_per_session": events_per_session,
        "avg_causal_connections": avg_connections_per_event,
    }


def sort_events_by_time(events):
    """Sort events chronologically by date."""
    return sorted(events, key=lambda e: date.fromisoformat(e["date"]))


def get_random_time():
    """Generate a random time delta between 08:00 and 22:00."""
    return timedelta(hours=random.randint(8, 21), minutes=random.randint(0, 59))


def get_session_dates(start_date, num_days, num_sessions):
    """Generate a list of random, sorted dates for conversation sessions."""
    possible_dates = [start_date + timedelta(days=i) for i in range(num_days)]
    if num_sessions > len(possible_dates):
        logging.warning(
            "Number of sessions is greater than number of days. Using all possible dates."
        )
        return sorted(possible_dates)
    return sorted(random.sample(possible_dates, num_sessions))


def get_relevant_events(events, current_date, last_session_date=None, lookback_days=7):
    """
    Filter events that occurred between the last session and the current one.
    Enhanced with lookback window to handle gaps between events.
    Based on LOCOMO's get_relevant_events with improvements.

    Args:
        events: List of all events
        current_date: Current session date
        last_session_date: Previous session date (None for first session)
        lookback_days: Days to look back if no new events (default 7)

    Returns:
        List of relevant events with causal context enrichment
    """
    # Sort events chronologically first (LOCOMO approach)
    sorted_events = sort_events_by_time(events)

    if last_session_date is None:
        # First session: include all events up to current date
        relevant = [
            e for e in sorted_events if date.fromisoformat(e["date"]) <= current_date
        ]
    else:
        # Subsequent sessions: events between last session and now
        new_events = [
            e
            for e in sorted_events
            if last_session_date < date.fromisoformat(e["date"]) <= current_date
        ]

        # If no new events, use lookback window to include recent context
        if not new_events:
            lookback_date = current_date - timedelta(days=lookback_days)
            relevant = [
                e
                for e in sorted_events
                if lookback_date <= date.fromisoformat(e["date"]) <= current_date
            ]
            if relevant:
                logging.info(
                    f"📅 No new events since last session. Using {len(relevant)} recent events from lookback window ({lookback_days} days)"
                )
        else:
            relevant = new_events

    # For each relevant event, include its causal context (what caused it)
    # This helps the conversation be more coherent
    event_map = {e["id"]: e for e in sorted_events}
    enriched_events = []

    for event in relevant:
        enriched_event = event.copy()
        # Add descriptions of causal events for context
        if event.get("caused_by"):
            causal_context = []
            for cause_id in event["caused_by"]:
                if cause_id in event_map:
                    causal_context.append(
                        f"{cause_id}: {event_map[cause_id]['description']}"
                    )
            if causal_context:
                enriched_event["causal_context"] = (
                    " (dipicu oleh: " + ", ".join(causal_context) + ")"
                )
        enriched_events.append(enriched_event)

    return enriched_events


def generate_conversation_session(
    user_profile,
    secondary_personas,
    session_id,
    current_date,
    recent_events,
    last_session_date,
    conversation_summary,
    min_turns,
    max_turns,
    cached_content=None,
):
    """Generate a conversation session using Gemini."""
    logging.info(
        "Generating session %d (Date: %s)...",
        session_id,
        current_date.strftime("%Y-%m-%d"),
    )

    # Format inputs
    secondary_personas_str = ", ".join(
        [f"{p['name']} ({p['relationship']})" for p in secondary_personas]
    )

    recent_events_str = ""
    if recent_events:
        for e in recent_events:
            recent_events_str += f"- [{e['date']}] {e['description']}\n"
    else:
        recent_events_str = "Tidak ada kejadian khusus baru-baru ini."

    last_datetime_str = (
        last_session_date.strftime("%Y-%m-%d") if last_session_date else "awal mula"
    )
    current_datetime_str = current_date.strftime("%Y-%m-%d %H:%M")

    # Construct prompt based on caching availability
    if cached_content:
        # Use ONLY the dynamic part, as static part is cached
        prompt = CONVERSATION_DYNAMIC_PROMPT.format(
            conversation_summary=conversation_summary,
            recent_events=recent_events_str,
            last_datetime_str=last_datetime_str,
            current_datetime_str=current_datetime_str,
            session_id=session_id,
        )
        logging.info("Using Context Caching for session generation.")
    else:
        # Combine Static + Dynamic for full prompt
        full_template = CONVERSATION_STATIC_PROMPT + "\n" + CONVERSATION_DYNAMIC_PROMPT
        prompt = full_template.format(
            user_name=user_profile["name"],
            user_profile=json.dumps(user_profile, indent=2),
            secondary_personas=secondary_personas_str,
            num_turns=random.randint(min_turns, max_turns),
            conversation_summary=conversation_summary,
            recent_events=recent_events_str,
            last_datetime_str=last_datetime_str,
            current_datetime_str=current_datetime_str,
            session_id=session_id,
            user_occupation=user_profile.get("occupation", "N/A"),
            user_location=user_profile.get("location", "N/A"),
        )
        logging.info("Using Standard Prompt (No Caching) for session generation.")

    max_retries = 6
    # min_turns is passed as argument

    for attempt in range(max_retries):
        logging.info(
            "Calling Gemini API for conversation (attempt %d/%d)...",
            attempt + 1,
            max_retries,
        )

        # HARD difficulty: Context-aware dialog
        # Pass cached_content if available
        response = run_gemini(
            prompt,
            max_output_tokens=8192,
            temperature=0.9,
            model_name="gemini-2.5-pro",
            cached_content=cached_content,
        )

        if not response:
            logging.error(
                "API failed to generate conversation for session %d (attempt %d/%d).",
                session_id,
                attempt + 1,
                max_retries,
            )
            continue

        # Parse the raw text into turns
        turns = []
        lines = response.strip().split("\n")
        for line in lines:
            if line.strip():
                match = re.match(r"^(Bot|User):\s*(.*)", line, re.IGNORECASE)
                if match:
                    speaker, text = match.groups()
                    turns.append({"speaker": speaker.lower(), "text": text.strip()})

        # Validate we got enough turns
        if len(turns) >= min_turns:
            logging.info("Generated %d turns for session %d.", len(turns), session_id)
            return turns
        else:
            logging.warning(
                "Generated only %d turns for session %d (attempt %d/%d), retrying...",
                len(turns),
                session_id,
                attempt + 1,
                max_retries,
            )

    logging.error(
        "Failed to generate sufficient turns for session %d after %d attempts.",
        session_id,
        max_retries,
    )
    return []


def generate_session_summary(user_name, session_turns):
    """Generate a summary for a conversation session."""
    if not session_turns:
        logging.warning("No turns provided for summary generation")
        return "Session kosong - tidak ada percakapan."

    # Limit conversation text to avoid safety filter
    conversation_text = "\n".join(
        [
            f"{turn['speaker'].capitalize()}: {turn['text']}"
            for turn in session_turns[:10]
        ]
    )  # Max 10 turns
    prompt = SESSION_SUMMARY_PROMPT.format(conversation_text=conversation_text)

    logging.info("Generating summary for session with %d turns...", len(session_turns))

    max_retries = 2
    for attempt in range(max_retries):
        # EASY difficulty: Simple summarization
        summary = run_gemini(
            prompt,
            max_output_tokens=512,
            temperature=0.5,
            model_name="gemini-2.5-flash-lite",
        )

        if summary:
            logging.info(
                "Summary generated successfully, length: %d characters", len(summary)
            )
            return summary.strip()
        else:
            logging.warning(
                "Failed to generate summary (attempt %d/%d) - trying simpler approach",
                attempt + 1,
                max_retries,
            )
            # Try even simpler prompt
            if attempt == 0:
                simple_prompt = f"Ringkas percakapan ini dalam 1-2 kalimat:\n{conversation_text[:500]}"
                summary = run_gemini(
                    simple_prompt,
                    max_output_tokens=256,
                    temperature=0.3,
                    model_name="gemini-2.5-flash-lite",
                )
                if summary:
                    return summary.strip()

    logging.warning("All summary attempts failed - using fallback")
    # Fallback: extract first and last user message
    user_turns = [t for t in session_turns if t["speaker"] == "user"]
    if len(user_turns) >= 2:
        return f"Percakapan tentang: {user_turns[0]['text'][:50]}... hingga {user_turns[-1]['text'][:50]}..."
    elif user_turns:
        return f"Percakapan singkat tentang: {user_turns[0]['text'][:100]}..."
    else:
        return "Percakapan antara bot dan user."


# --- Ground Truth Generation Functions ---


def generate_ground_truth_for_turn(turn, session_date, user_profile, session_id):
    """Generate ground truth annotations untuk satu turn percakapan"""

    prompt = f"""Kamu adalah annotator untuk dataset evaluasi RAG system.

User Profile:
- Name: {user_profile["name"]}
- Occupation: {user_profile["occupation"]}

Session Date: {session_date}
Speaker: {turn["speaker"]}
Text: {turn["text"]}

Tugasmu adalah extract informasi berikut dalam format JSON:

1. **factual**: List of factual statements/assertions yang bisa diverifikasi
   - fact: the factual statement
   - confidence: high/medium/low
   - category: personal_info/event/preference/plan/opinion

2. **entities_mentioned**: List of entities disebutkan
   - name: nama entity
   - type: person/place/organization/time/other
   - context: bagaimana entity disebutkan

3. **temporal_references**: List of temporal info
   - type: absolute/relative
   - value: nilai waktu
   - text: teks asli yang menyebut waktu

4. **retrieval_required**: Informasi apa yang perlu di-retrieve dari history untuk memahami turn ini
   - info_type: previous_event/user_trait/relationship/context
   - description: deskripsi info yang dibutuhkan

Output harus valid JSON. Jika tidak ada data, return empty array [].
HANYA return JSON, tidak ada text lain.
"""

    try:
        # EASY difficulty: Information extraction per turn
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

        model = GenerativeModel("gemini-2.5-flash-lite")
        response = model.generate_content(
            prompt,
            generation_config={"temperature": 0.3, "max_output_tokens": 4096},
            safety_settings=safety_settings,
        )

        if not response or not response.text:
            return None

        json_match = re.search(r"\{[\s\S]*\}", response.text)
        if json_match:
            return json.loads(json_match.group())
        return None

    except Exception as e:
        logging.error("Error generating GT: %s", str(e))
        return None


def resolve_ground_truth_conflicts(existing_facts, new_facts, session_context):
    """
    Use LLM to resolve conflicts between existing and new ground truth facts.
    LLM determines which facts are still valid, which are superseded, and how to merge them.

    Args:
        existing_facts: List of all existing facts from previous sessions
        new_facts: New facts from current session
        session_context: Context about current session (date, summary)

    Returns:
        Resolved facts with conflict resolution metadata
    """
    if not existing_facts or not new_facts:
        return new_facts

    # Build context for LLM
    existing_facts_str = json.dumps(
        existing_facts[:50], indent=2, ensure_ascii=False
    )  # Limit to last 50 facts
    new_facts_str = json.dumps(new_facts, indent=2, ensure_ascii=False)

    prompt = f"""Kamu adalah sistem yang mengelola ground truth facts dalam knowledge graph temporal.

EXISTING FACTS (dari percakapan sebelumnya):
{existing_facts_str}

NEW FACTS (dari percakapan terbaru pada {session_context["date"]}):
{new_facts_str}

TUGAS:
Identifikasi jika ada konflik antara existing facts dan new facts, kemudian return semua facts (existing + new) yang masih valid.

ATURAN:
- Jika new fact bertentangan dengan existing fact (misal: jadwal dirubah, status berubah), gunakan new fact
- Jika new fact melengkapi existing fact, keep both
- Jika tidak ada konflik, keep both
- Return semua facts yang masih valid dalam array "resolved_facts"

OUTPUT (JSON only, no explanation):
{{
  "resolved_facts": [
    // Array of all valid facts (existing + new)
  ]
}}"""

    try:
        # Use fast model for conflict resolution (gemini-2.5-flash-lite)
        resolution_text = run_gemini(
            prompt,
            max_output_tokens=8192,
            temperature=0.2,  # Low temperature for consistent logic
            model_name="gemini-2.5-flash-lite",
        )

        if not resolution_text:
            logging.warning("  ⚠️  Empty response from conflict resolution API")
            return new_facts

        # Try to parse JSON response
        resolution_text = resolution_text.strip()

        # Remove markdown code blocks if present
        if resolution_text.startswith("```json"):
            resolution_text = (
                resolution_text.split("```json")[1].split("```")[0].strip()
            )
        elif resolution_text.startswith("```"):
            resolution_text = resolution_text.split("```")[1].split("```")[0].strip()

        resolution = json.loads(resolution_text)

        # Log conflicts found (if any were actually resolved)
        resolved_facts = resolution.get("resolved_facts", [])
        if resolved_facts:
            # Check if there were actual resolutions (resolved_facts != existing + new)
            num_resolved = len(resolved_facts)
            num_expected = len(existing_facts[:50]) + len(
                new_facts
            )  # Expected if no conflicts
            if num_resolved < num_expected:
                num_conflicts = num_expected - num_resolved
                logging.info(
                    "  🔄 Resolved %d conflicts (merged/superseded facts)",
                    num_conflicts,
                )

        return resolved_facts if resolved_facts else new_facts

    except Exception as e:
        logging.warning(
            "  ⚠️  Conflict resolution failed: %s. Using new facts as-is.", str(e)
        )
        return new_facts


def generate_ground_truth_annotations(
    conversation_history, events, output_dir, incremental=False
):
    """
    Generate ground truth annotations for sessions.

    Args:
        conversation_history: Dataset with sessions
        events: User events
        output_dir: Output directory
        incremental: If True, skip sessions with existing ground_truths
    """

    logging.info("=" * 80)
    logging.info("GENERATING GROUND TRUTH ANNOTATIONS")
    logging.info("=" * 80)

    sessions = conversation_history["sessions"]
    user_profile = conversation_history["user"]
    total_turns = sum(len(s["turns"]) for s in sessions)
    total_annotated = 0

    # Collect all existing facts for conflict resolution
    all_existing_facts = []
    if incremental:
        for session in sessions:
            if "ground_truths" in session:
                for gt in session["ground_truths"]:
                    all_existing_facts.extend(gt.get("factual", []))

        if all_existing_facts:
            logging.info(
                "📚 Loaded %d existing facts for conflict resolution",
                len(all_existing_facts),
            )

    for session in sessions:
        session_id = session["session_id"]
        session_date = session["date"]

        # INCREMENTAL: Skip if session already has ground_truths
        if incremental and "ground_truths" in session and session["ground_truths"]:
            logging.info(
                "⏭️  Session %d (%s) - SKIPPED (already annotated)",
                session_id,
                session_date,
            )
            total_annotated += len(session["ground_truths"])
            continue

        logging.info(
            "Session %d (%s) - %d turns",
            session_id,
            session_date,
            len(session["turns"]),
        )

        ground_truths = []
        session_new_facts = []

        for turn_idx, turn in enumerate(session["turns"]):
            gt = generate_ground_truth_for_turn(
                turn, session_date, user_profile, session_id
            )
            time.sleep(DELAY_BETWEEN_REQUESTS)

            if gt:
                # Collect new facts for conflict resolution
                session_new_facts.extend(gt.get("factual", []))

                gt_entry = {
                    "turn_id": turn_idx,
                    "session_id": session_id,
                    "speaker": turn["speaker"],
                    **gt,
                }
                ground_truths.append(gt_entry)
                total_annotated += 1
                logging.info(
                    "  ✓ Turn %d: %d facts, %d entities",
                    turn_idx,
                    len(gt.get("factual", [])),
                    len(gt.get("entities_mentioned", [])),
                )
            else:
                logging.warning("  ⚠️  Turn %d: Failed", turn_idx)

        # CONFLICT RESOLUTION: If incremental and has existing facts, resolve conflicts
        if incremental and all_existing_facts and session_new_facts:
            session_context = {
                "date": session_date,
                "session_id": session_id,
                "summary": session.get("summary", ""),
            }

            logging.info(
                "  🔍 Checking for conflicts with %d existing facts...",
                len(all_existing_facts),
            )
            resolved_facts = resolve_ground_truth_conflicts(
                all_existing_facts, session_new_facts, session_context
            )

            # Update ground_truths with resolved facts
            if resolved_facts != session_new_facts:
                # Redistribute resolved facts back to turns
                fact_idx = 0
                for gt in ground_truths:
                    num_facts = len(gt.get("factual", []))
                    if fact_idx < len(resolved_facts):
                        gt["factual"] = resolved_facts[fact_idx : fact_idx + num_facts]
                        fact_idx += num_facts

            # Add resolved facts to existing pool
            all_existing_facts.extend(resolved_facts)

        session["ground_truths"] = ground_truths
        logging.info("✓ Session %d: %d turns annotated", session_id, len(ground_truths))

    logging.info("Total: %d/%d turns annotated", total_annotated, total_turns)

    # Generate evaluation queries
    logging.info("=" * 80)
    logging.info("GENERATING EVALUATION QUERIES")
    logging.info("=" * 80)

    time.sleep(DELAY_BETWEEN_REQUESTS)
    # TODO: implement generate_evaluation_queries function
    queries: list = []  # generate_evaluation_queries(sessions, user_profile, events)
    conversation_history["evaluation_queries"] = queries
    logging.info("✓ Generated %d evaluation queries", len(queries))

    # Save to main dataset file (no separate enhanced file)
    main_output = os.path.join(output_dir, "conversation_dataset.json")
    with open(main_output, "w", encoding="utf-8") as f:
        json.dump(conversation_history, f, indent=2, ensure_ascii=False)

    logging.info("✓ Dataset with ground truth saved: %s", main_output)
    logging.info("=" * 80)
    logging.info("✅ GROUND TRUTH GENERATION COMPLETE!")
    logging.info("=" * 80)

    return main_output


def main():
    args = parse_args()

    set_gemini_key()
    os.makedirs(args.out_dir, exist_ok=True)

    # Set token log path to be inside the output directory
    set_token_log_path(args.out_dir)

    conversation_history = {}
    events = []
    start_session_id = 1
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d").date()
    _last_datetime = None
    cumulative_summary = ""
    output_file = os.path.join(args.out_dir, "conversation_dataset.json")
    events_file = os.path.join(args.out_dir, "user_events.json")

    user_profile = None
    secondary_personas = []

    # Load or generate user profile FIRST (before checking existing dataset)
    if args.auto_generate_persona:
        logging.info("🎲 Auto-generate persona mode enabled")
        persona_data = generate_random_persona()

        if persona_data:
            user_profile = persona_data["user"]
            secondary_personas = persona_data["secondary_personas"]

            # Save generated persona to output directory for reference
            temp_persona_file = os.path.join(
                args.out_dir, "auto_generated_persona.json"
            )
            with open(temp_persona_file, "w", encoding="utf-8") as f:
                json.dump(persona_data, f, indent=2, ensure_ascii=False)
            logging.info("✅ Saved auto-generated persona to: %s", temp_persona_file)
        else:
            logging.warning(
                "Auto-generation failed, falling back to default file: %s",
                args.user_file,
            )
            user_profile, secondary_personas = load_user_profile(args.user_file)
    # If not auto-generating, will load from file later (after checking existing dataset)

    # Auto-detect existing dataset for incremental generation
    existing_dataset_path = output_file
    if os.path.exists(existing_dataset_path) and not args.fresh_start:
        logging.info("Detected existing dataset at: %s", existing_dataset_path)
        logging.info("Will add %d NEW sessions incrementally...", args.num_sessions)

        try:
            with open(existing_dataset_path, "r", encoding="utf-8") as f:
                conversation_history = json.load(f)

            if os.path.exists(events_file):
                with open(events_file, "r", encoding="utf-8") as f:
                    events = json.load(f)

            user_profile = conversation_history["user"]
            secondary_personas = conversation_history.get("secondary_personas", [])

            # Initialize defaults
            last_session_date = start_date

            if conversation_history["sessions"]:
                last_session = conversation_history["sessions"][-1]
                start_session_id = last_session["session_id"] + 1
                last_session_date = date.fromisoformat(last_session["date"])
                _last_datetime = datetime.strptime(
                    last_session["datetime"], "%d %B %Y, %H:%M"
                )
                start_date = last_session_date + timedelta(
                    days=1
                )  # Continue from next day

            cumulative_summary = conversation_history.get("summary", "")
            logging.info(
                "Successfully loaded existing data. Will generate sessions %d-%d.",
                start_session_id,
                start_session_id + args.num_sessions - 1,
            )

            # === INCREMENTAL EVENT GENERATION ===
            logging.info("=" * 60)
            logging.info("🎲 INCREMENTAL EVENT GENERATION MODE")
            logging.info("=" * 60)

            # Calculate metrics from existing data
            logging.info(
                f"📊 Existing: {len(conversation_history['sessions'])} sessions"
            )
            logging.info(f"📊 Existing: {len(events)} events")

            # Determine how many new events we need
            # Rule: ~0.8 event per session for high density (matching initial ratio)
            new_events_needed = max(int(args.num_sessions * 0.8), 5)  # Min 5 events

            # Calculate date range for new events
            # Based on existing session density
            existing_sessions = len(conversation_history["sessions"])
            existing_span = (
                last_session_date
                - date.fromisoformat(conversation_history["sessions"][0]["date"])
            ).days
            session_density = (
                existing_span / existing_sessions if existing_sessions > 0 else 2
            )
            new_span_days = int(
                args.num_sessions * session_density
            )  # Proportional to session count

            # Get last event info
            last_event_date = (
                max([date.fromisoformat(e["date"]) for e in events])
                if events
                else start_date
            )
            last_event_id = max([int(e["id"][1:]) for e in events]) if events else 0

            new_events = []
            if new_events_needed > 0:
                logging.info(
                    f"⚡ Generating {new_events_needed} additional events for density..."
                )
                new_events = generate_events_continue(
                    user_profile,
                    secondary_personas,
                    events,
                    last_event_id,
                    last_event_date,
                    new_span_days,
                    new_events_needed,
                )

            if new_events:
                logging.info(f"✅ Generated {len(new_events)} new events")

                # Merge with existing events
                events.extend(new_events)
                logging.info(f"📊 Total events after merge: {len(events)}")

                # Filter standalone events to maintain dense causal connections
                logging.info(
                    "🔍 Filtering standalone events... (DISABLED for richer dataset)"
                )
                # events = filter_standalone_events(events)

                # Save updated events
                with open(events_file, "w", encoding="utf-8") as f:
                    json.dump(events, f, indent=2, ensure_ascii=False)
                logging.info(f"💾 Saved updated events to {events_file}")

                # Log final event IDs
                event_ids = [e["id"] for e in events]
                logging.info(f"📋 Event IDs: {event_ids[0]} to {event_ids[-1]}")
            else:
                logging.warning(
                    "⚠️ No new events generated. Will use existing events only."
                )

            logging.info("=" * 60)

        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            logging.error(
                "Failed to load existing dataset. Error: %s. Starting fresh.", e
            )
            args.fresh_start = True

    if args.fresh_start or not os.path.exists(existing_dataset_path):
        logging.info("Starting FRESH dataset generation...")
        # Load user profile if not already loaded (from auto-generate)
        if user_profile is None:
            user_profile, secondary_personas = load_user_profile(args.user_file)

        # Smart Event Sizing: Use user arg OR calculate based on session count (0.8 ratio)
        # This ensures fresh start with 100 sessions gets ~80 events immediately
        smart_num_events = max(args.num_events, int(args.num_sessions * 0.8))
        logging.info(
            f"🧠 Smart Sizing: Generating {smart_num_events} initial events for {args.num_sessions} sessions."
        )

        events = generate_events(
            user_profile,
            secondary_personas,
            start_date,
            args.num_days,
            smart_num_events,
        )

        with open(events_file, "w", encoding="utf-8") as f:
            json.dump(events, f, indent=2, ensure_ascii=False)
        logging.info("Saved %d events to %s", len(events), events_file)

        conversation_history = {
            "user": user_profile,
            "secondary_personas": secondary_personas,
            "sessions": [],
            "summary": "",
        }

    # Ensure user_profile is set (for incremental generation without auto-generate)
    if user_profile is None:
        user_profile, secondary_personas = load_user_profile(args.user_file)

    # Generate new session dates, starting from the determined start_date
    new_session_dates = get_session_dates(start_date, args.num_days, args.num_sessions)

    # --- Automatic Event Scaling (Ensure 0.8 events/session) ---
    total_sessions_planned = (start_session_id - 1) + args.num_sessions
    target_num_events = int(total_sessions_planned * 0.8)

    # Ensure minimum 20 events or user specified num_events if higher
    target_num_events = max(target_num_events, args.num_events)

    current_event_count = len(events)
    events_needed = target_num_events - current_event_count

    if events_needed > 0:
        logging.info("=" * 60)
        logging.info(
            f"⚖️  EVENT BALANCING: Found {current_event_count} events for {total_sessions_planned} sessions."
        )
        logging.info(
            f"   Target density 0.8/session requires {target_num_events} events."
        )
        logging.info(f"   Generating {events_needed} NEW events to maintain quality...")
        logging.info("=" * 60)

        batch_size = 10
        num_batches = (events_needed + batch_size - 1) // batch_size

        for i in range(num_batches):
            current_batch_size = min(batch_size, events_needed - (i * batch_size))
            logging.info(
                f"🔄 Batch {i + 1}/{num_batches}: Generating {current_batch_size} events..."
            )

            # Determine parameters for continuation
            last_event = events[-1] if events else None
            last_event_id = 0
            if last_event:
                try:
                    last_event_id = int(last_event["id"][1:])
                except ValueError:
                    last_event_id = len(events)

            # Generate batch
            new_batch = generate_events_continue(
                user_profile,
                secondary_personas,
                events,
                last_event_id,
                start_date,
                args.num_days,
                current_batch_size,
            )

            if new_batch:
                events.extend(new_batch)
                # Save intermediate progress
                with open(events_file, "w", encoding="utf-8") as f:
                    json.dump(events, f, indent=2, ensure_ascii=False)
                logging.info(
                    f"   ✅ Added {len(new_batch)} events. Total: {len(events)}"
                )
            else:
                logging.warning(
                    "   ⚠️ Failed to generate batch. Stopping event generation."
                )
                break

        logging.info(f"🎉 Event generation complete. Final count: {len(events)}")
        logging.info("=" * 60)
    else:
        logging.info(
            f"✅ Event count ({current_event_count}) is sufficient for {total_sessions_planned} sessions (Target: {target_num_events})."
        )

    # --- Cache Creation (if enabled) ---
    cached_content = None
    if args.use_caching:
        logging.info("🚀 Initializing Context Caching for Gemini 2.5...")
        secondary_personas_str = ", ".join(
            [f"{p['name']} ({p['relationship']})" for p in secondary_personas]
        )

        # Format the static prompt
        # Note: We use a fixed range for num_turns in the cached prompt to avoid invalidating cache
        static_prompt_content = CONVERSATION_STATIC_PROMPT.format(
            user_name=user_profile["name"],
            user_profile=json.dumps(user_profile, indent=2),
            secondary_personas=secondary_personas_str,
            num_turns=f"{args.min_turns_per_session}-{args.max_turns_per_session}",
            user_occupation=user_profile.get("occupation", "N/A"),
            user_location=user_profile.get("location", "N/A"),
        )

        try:
            # Create cache with 60 minute TTL
            cached_content = caching.CachedContent.create(
                model="models/gemini-2.5-pro",
                display_name="locomo_static_context",
                system_instruction=static_prompt_content,
                ttl=timedelta(minutes=60),
            )
            logging.info(f"✅ Cache created: {cached_content.name} (TTL: 60 mins)")
        except Exception as e:
            logging.error(f"Failed to create cache: {e}")
            logging.warning("Falling back to standard generation without caching.")
            cached_content = None

    # --- Generation Loop for New Sessions ---
    for i, session_date in enumerate(new_session_dates):
        session_id = start_session_id + i

        session_time = get_random_time()
        current_datetime = (
            datetime.combine(session_date, datetime.min.time()) + session_time
        )
        current_datetime_str = current_datetime.strftime("%d %B %Y, %H:%M")

        last_session_date = (
            date.fromisoformat(conversation_history["sessions"][-1]["date"])  # type: ignore[invalid-argument-type]
            if conversation_history["sessions"]
            else None
        )
        relevant_events = get_relevant_events(events, session_date, last_session_date)

        session_turns = generate_conversation_session(
            user_profile=user_profile,
            secondary_personas=secondary_personas,
            session_id=session_id,
            current_date=current_datetime,
            recent_events=relevant_events,
            last_session_date=last_session_date,
            conversation_summary=conversation_history.get("summary", ""),
            min_turns=args.min_turns_per_session,
            max_turns=args.max_turns_per_session,
            cached_content=cached_content,
        )

        session_summary = generate_session_summary(user_profile["name"], session_turns)

        session_data = {
            "session_id": session_id,
            "date": session_date.strftime("%Y-%m-%d"),
            "datetime": current_datetime_str,
            "turns": session_turns,
            "summary": session_summary,
            "relevant_events": relevant_events,
        }
        conversation_history["sessions"].append(session_data)  # type: ignore[possibly-missing-attribute,invalid-argument-type]

        if cumulative_summary:
            cumulative_summary += (
                f"\n\nSesi {session_id} ({session_date}): {session_summary}"
            )
        else:
            cumulative_summary = (
                f"Sesi {session_id} ({session_date}): {session_summary}"
            )

        conversation_history["summary"] = cumulative_summary
        _last_datetime = current_datetime

        logging.info("Completed and saved session %d.", session_id)
        print(f"\n{'=' * 60}\n")

        # Save progress after each session
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(conversation_history, f, indent=2, ensure_ascii=False)

    logging.info("Saved complete dataset to %s", output_file)

    # Calculate and log event density metrics
    calculate_event_density(events, conversation_history["sessions"])

    total_turns = sum(len(s["turns"]) for s in conversation_history["sessions"])  # type: ignore[invalid-argument-type,not-iterable]
    separator = "=" * 60
    print(f"\n{separator}")
    print("DATASET GENERATION COMPLETED")
    print(separator)
    print(f"User: {user_profile['name']}")
    print(f"Total sessions in file: {len(conversation_history['sessions'])}")
    print(f"New sessions generated: {args.num_sessions}")
    print(f"Total turns in file: {total_turns}")
    print(
        f"Date range: {conversation_history['sessions'][0]['date']} to {conversation_history['sessions'][-1]['date']}"  # type: ignore[invalid-argument-type]
    )
    print(f"Total events: {len(events)}")
    print(f"Output saved to: {output_file}")
    print(f"{separator}\n")

    # Detect if we're in incremental mode (existing dataset was loaded)
    is_incremental = start_session_id > 1

    # Always generate ground truth annotations after dataset generation
    logging.info("\n🎯 Starting ground truth generation...")
    if is_incremental:
        logging.info("📝 Incremental mode: Will skip re-annotating existing sessions")

    output_with_gt = generate_ground_truth_annotations(
        conversation_history, events, args.out_dir, incremental=is_incremental
    )

    print(f"\n{separator}")
    print("GROUND TRUTH GENERATION COMPLETED")
    print(separator)
    print(f"Dataset with ground truth: {output_with_gt}")
    print(f"Run validation: python test_enhanced_dataset.py {output_with_gt}")
    print(f"{separator}\n")


if __name__ == "__main__":
    main()
