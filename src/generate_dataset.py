import os
import sys
import random
import pandas as pd
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import set_seed, ensure_directories, RAW_DATA_DIR

# Set seed for deterministic generation
set_seed(42)
ensure_directories()

# ==========================================
# 1. EXPANDED TEMPLATES & VOCABULARY FOR NORMAL
# ==========================================

NORMAL_TEMPLATES = [
    "Starting my {time_of_day} with a warm cup of {drink} and some peaceful {genre} music.",
    "Just finished cleaning the entire {room}. Feels so refreshing to have an organized space!",
    "Woke up earlier than usual today to catch the {phenomenon} over {nature_spot}. Absolutely breathtaking.",
    "Heading out for a quick {activity} before the afternoon rush starts in {city}.",
    "Spent the afternoon organizing my bookshelf. Rediscovered so many classic {genre} books from {author_era}.",
    "A quiet evening at home with {snack} and my favorite {media_type}.",
    "Laundry day is finally done! Now time to relax and recharge for the {day_of_week}.",
    "Enjoying a slow {day_of_week} breakfast with freshly baked {food_item} and {drink}.",
    "Had such a good night's sleep. Ready to tackle everything on my to-do list for {day_of_week}!",
    "Taking a walk around {nature_spot} to get some fresh air and clear my head after work.",
    "Exploring the historic streets of {city}. The architecture here in {city_quarter} is unbelievable!",
    "Hiking up the scenic trail at {nature_spot} was tough, but the view from the summit was worth every step.",
    "Spent the weekend camping near {water_body}. The night sky was full of stars and {astronomy_item}.",
    "Road trip across the countryside toward {city} with good friends and great playlists.",
    "Caught a stunning sunset by {landscape} today. Nature never fails to amaze me.",
    "Visiting the local botanical garden in {city}. The {flower} collection is in full bloom!",
    "Packing bags for our upcoming trip to {destination}. Can't wait to explore the local culture and try {cuisine_dish}.",
    "Sitting by the beach near {water_body} listening to the gentle ocean waves. Pure relaxation.",
    "The mountain breeze this {time_of_day} feels so crisp and revitalizing while walking near {nature_spot}.",
    "Found a cozy hidden cafe tucked away in an old cobblestone alley in {city}.",
    "Tried cooking homemade {cuisine_dish} from scratch for the first time! Turned out delicious with fresh {herb}.",
    "Baking fresh {pastry} always makes the entire kitchen smell like heaven on a {day_of_week}.",
    "Had lunch at this cozy little {cuisine_type} diner in downtown {city}. Highly recommend the {specialty}!",
    "Experimenting with a new pasta recipe tonight with garlic, olive oil, and fresh {herb}.",
    "Nothing beats a warm bowl of hearty {soup_type} on a chilly {day_of_week} evening.",
    "Made a refreshing smoothie bowl with fresh berries, bananas, and {topping}.",
    "Sunday family barbecue in the backyard was a huge success today with grilled {food_item}.",
    "Trying out local street food stalls in {city}. The authentic {cuisine_type} flavors here are incredible.",
    "Meal prepping healthy lunches for the upcoming work week feels so satisfying and organized.",
    "Freshly brewed artisanal {coffee_type} to kick off a productive {day_of_week} morning.",
    "Just completed an insightful online course on {tech_topic}. Learned so much about modern software architectures.",
    "Studying for upcoming exams in {subject}. Making flashcards and reviewing key theorems.",
    "Finally solved that tricky bug in my {programming_lang} project after hours of careful debugging!",
    "Reading a fascinating research paper regarding recent developments in {science_field}.",
    "Attended an inspiring virtual tech conference on sustainable software engineering and {tech_topic} today.",
    "Setting up my new development workstation in the {room}. Clean cable management makes a huge difference.",
    "Working on a collaborative group assignment with classmates for our {course_name} module at university.",
    "Learning the fundamentals of data structures and algorithms in {programming_lang} is actually really rewarding.",
    "Taking detailed notes during today's guest lecture on ethical considerations in engineering.",
    "Built a simple web application using modern {tech_topic} tools. Progress feels fantastic!",
    "Just watched the season finale of {tv_show}. That unexpected cliffhanger was mind-blowing!",
    "Practicing chords on my {instrument} for an hour every {day_of_week} is finally paying off.",
    "Spent the weekend playing {game_title} with friends online. Such great multiplayer fun!",
    "Listening to the new album by {artist_genre} on repeat. The acoustic production quality is outstanding.",
    "Attended a vibrant live concert in downtown {city} yesterday evening. The crowd energy was electric.",
    "Finished sketching a detailed portrait in my art journal. Experimenting with charcoal shading and perspective.",
    "Cheering for our university team during the {sport} championship match today in {city}!",
    "Visited the modern art museum exhibition. Loved the creative use of geometric textures and lighting.",
    "Board game night with family. Playing {game_title} always brings out everyone's competitive side!",
    "Photography walk through {city_quarter} capturing urban architecture and candid street scenes.",
    "Does anyone have good recommendations for a reliable wireless mechanical keyboard for coding?",
    "What is your favorite book that completely changed your perspective on {subject}?",
    "Friendly reminder to stay hydrated, take regular screen breaks, and stretch throughout your {day_of_week}.",
    "Personally, I prefer reading physical paperback books over digital e-readers when traveling. How about you?",
    "What are your go-to productivity tips when working remotely from home on {tech_topic}?",
    "Looking for beginner-friendly podcast recommendations on world history, art, and {science_field}.",
    "Celebrating small victories every day keeps you motivated in the long run during difficult projects.",
    "Grateful for supportive friends and mentors who always encourage personal and professional growth.",
    "Which season of the year do you enjoy most: autumn foliage in {nature_spot} or spring blossoms in {city}?",
    "Always be kind to service workers and delivery drivers. A little courtesy and a warm smile go a long way."
]

NORMAL_SLOTS = {
    "time_of_day": ["morning", "afternoon", "early dawn", "evening", "weekend morning", "sunrise hour"],
    "day_of_week": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday", "weekend"],
    "drink": ["green tea", "cappuccino", "hot cocoa", "earl grey tea", "fresh orange juice", "iced latte", "chai", "herbal tea", "matcha latte", "peppermint tea", "chamomile infusion", "cold brew coffee"],
    "room": ["study room", "living room", "kitchen", "garage workspace", "home office", "bedroom closet", "studio apartment", "reading nook", "balcony garden"],
    "phenomenon": ["golden sunrise", "morning mist", "dew on spring flowers", "clear blue dawn", "early birds singing", "gentle rain shower", "warm morning sunlight"],
    "activity": ["jog in the park", "cycling session", "yoga routine", "stretch session", "gym workout", "morning stroll", "brisk walk", "pilates workout"],
    "genre": ["science fiction", "historical fiction", "biography", "philosophy", "classic literature", "adventure novel", "mystery novel", "poetry", "nature documentary"],
    "author_era": ["the 19th century", "the Renaissance", "modern literature", "the golden age of sci-fi", "contemporary writers", "ancient philosophy"],
    "snack": ["buttered popcorn", "dark chocolate", "roasted almonds", "fruit salad", "warm cookies", "herbal tea", "salted cashews", "banana slices", "trail mix"],
    "media_type": ["documentary series", "favorite comedy sitcom", "audiobook", "jazz playlist", "podcast episode", "nature stream", "indie film", "acoustic session"],
    "food_item": ["cinnamon rolls", "sourdough bread", "blueberry pancakes", "waffles with honey", "french toast", "fresh croissants", "banana muffins", "bagels with cream cheese"],
    "city": ["Kyoto", "Edinburgh", "Barcelona", "Florence", "Vancouver", "Prague", "Vienna", "Melbourne", "Boston", "Seattle", "Dublin", "Zurich", "Toronto", "Stockholm", "San Francisco"],
    "city_quarter": ["the Old Town", "the Arts District", "the Riverside Quarter", "the Historic Center", "the Waterfront Promenade", "the Harbor District"],
    "nature_spot": ["Pine Ridge Mountain", "Silver Falls Park", "Emerald Valley", "Blue Ridge Crest", "Cascade Canyon", "Highland Meadows", "Sunset Bluff", "Maple Forest"],
    "water_body": ["crystal mountain lake", "coastal bay", "quiet river bend", "alpine reservoir", "serene pond", "ocean cove", "emerald stream"],
    "astronomy_item": ["the Milky Way band", "shooting meteors", "the bright moon", "distant constellations", "the North Star"],
    "landscape": ["cliffside overlook", "seaside pier", "rolling green hills", "desert horizon", "lakeside dock", "alpine valley", "forest canopy"],
    "flower": ["wild orchid", "spring tulip", "cherry blossom", "lavender", "rose garden", "sunflower", "daisy", "magnolia bloom"],
    "destination": ["the Swiss Alps", "the Mediterranean coast", "the Pacific Northwest", "the Scandinavian fjords", "the Scottish Highlands", "the Japanese Alps"],
    "cuisine_dish": ["lasagna", "vegetable stir-fry", "chicken tikka masala", "mushroom risotto", "ramen noodles", "street tacos", "ratatouille", "gnocchi", "falafel wrap"],
    "pastry": ["chocolate chip cookies", "banana walnut bread", "apple cinnamon tart", "butter croissants", "fudgy brownies", "blueberry scones", "lemon loaf"],
    "cuisine_type": ["Italian", "Japanese", "Mediterranean", "Mexican", "Thai", "French", "Indian", "Vietnamese", "Greek", "Korean"],
    "specialty": ["handmade pasta", "stone-baked pizza", "fresh sushi roll", "artisan sandwich", "grilled skewers", "crispy dumplings", "savory crepes"],
    "herb": ["basil and oregano", "rosemary and thyme", "fresh parsley", "cilantro and lime", "sage", "tarragon", "fresh dill"],
    "soup_type": ["roasted tomato soup", "creamy butternut squash soup", "lentil soup", "minestrone", "miso soup", "wild mushroom soup", "vegetable broth"],
    "topping": ["toasted chia seeds and honey", "granola and strawberries", "coconut flakes", "almond butter", "crushed walnuts", "fresh blueberries"],
    "coffee_type": ["pour-over roast", "espresso roast", "caramel macchiato", "french press blend", "cold brew", "hazelnut mocha", "dark roast blend"],
    "tech_topic": ["deep learning optimization", "cloud microservices", "distributed systems", "RESTful API design", "container orchestration", "database indexing", "neural networks", "software testing"],
    "subject": ["Linear Algebra", "Database Management", "Computer Networks", "Statistical Modeling", "Operating Systems", "Discrete Mathematics", "Data Structures"],
    "programming_lang": ["Python", "Rust", "TypeScript", "C++", "Go", "Java", "Kotlin", "Swift"],
    "science_field": ["renewable energy systems", "quantum computing basics", "marine biology", "astrophysics research", "environmental conservation", "computational biology"],
    "course_name": ["Software Engineering", "Artificial Intelligence", "Information Security", "Human-Computer Interaction", "Computer Architecture"],
    "tv_show": ["the latest sci-fi mystery", "the period drama", "the detective thriller", "the nature docuseries", "the animated series", "the culinary show"],
    "instrument": ["acoustic guitar", "piano keys", "violin", "electric bass", "ukulele", "classical cello", "keyboard synthesizer"],
    "game_title": ["an indie adventure game", "co-op puzzle games", "a space exploration sim", "strategy board games", "a retro platformer", "a sandbox builder"],
    "artist_genre": ["acoustic folk musicians", "classical orchestral ensembles", "lo-fi indie producers", "synthwave artists", "ambient jazz trios", "indie rock bands"],
    "sport": ["badminton tournament", "basketball regional", "soccer league", "track and field", "volleyball cup", "tennis championship", "cycling race"]
}

# ==========================================
# 2. EXPANDED TEMPLATES & VOCABULARY FOR THREAT
# ==========================================

THREAT_TEMPLATES = [
    # Category 1: Fake News / Fabricated Emergency / Disinformation
    "URGENT ALERT: Government officials in {location} secretly passed emergency order #{alert_id} banning all {resource} starting tomorrow midnight. Spread this before it gets censored!",
    "BREAKING DISCLOSURE: Whistleblower leaks verified internal dossier showing that {institution} intentionally contaminated municipal {infrastructure} in {location} with {toxic_agent}.",
    "EMERGENCY WARNING: Major satellite infrastructure failure predicted within {hour_count} hours will permanently disable the entire regional {network_type} network across {region}. Forward immediately!",
    "The mainstream media is deliberately suppressing the truth about the catastrophic {disaster_type} occurring in {location}. Over {victim_count} citizens are already being evacuated in secret!",
    "LEAKED AUDIO RECORDING: Senior health commissioner in {location} admits that recent {health_issue} was engineered in an offshore research facility to enforce mass lockdowns.",
    "FINANCIAL ALERT: All central and commercial banks in {region} will freeze private savings accounts starting {day_of_week} morning. Withdraw all your physical cash immediately!",
    "CONFIRMED REPORT: Secret government initiative #{alert_id} to force mandatory relocation of residents in {location} has officially commenced without public consultation.",
    "Do not drink the tap water or use municipal supply in {city_target}! Contaminated with lethal doses of {toxic_agent} following an undisclosed chemical plant explosion.",
    "Shocking classified intelligence: Top military official confirms hostile airborne drones carrying {toxic_agent} are entering civilian airspace over {region} tonight.",
    "They don't want the public to see this video: Absolute proof that elections in {location} were fabricated using counterfeit ballot machines and foreign code.",
    "URGENT PUBLIC NOTICE: Mandatory power blackout scheduled for {region} lasting {hour_count} hours to hide covert military operation in {location}.",
    "SECRET DIRECTIVE LEAKED: The authorities are preparing to confiscate private {resource} from homes across {location} starting this {day_of_week}.",
    "EXPOSED: Major pharmaceutical consortium and {institution} intentionally withheld antidote for {health_issue} to artificially inflate stock prices.",
    "CRITICAL WARNING: Water supply pipelines across {city_target} have been compromised with radioactive {toxic_agent}. Do not trust official news statements!",
    "MASSIVE COVERUP: Hospital intensive care units in {location} are completely overwhelmed due to secret deployment of {toxic_agent}. Media ordered to remain silent!",

    # Category 2: Safe Synthetic Hate Speech / Targeted Hostility / Harassment
    "People belonging to {group_target} are a biological menace to our culture and should be forcefully expelled from {location} immediately.",
    "We need to systematically drive {group_target} out of our neighborhood in {city_target}. They are subhuman pests ruining community safety.",
    "Hey @{user_tag}, you are an absolute piece of garbage and you don't deserve to breathe our air. Watch your back wherever you go in {city_target}.",
    "Targeted warning to all {group_target}: Get out of our city or face severe retaliation from local militant groups.",
    "This disgusting person @{user_tag} should be publicly humiliated and hunted down. You will pay for what you posted.",
    "All members of {group_target} are inherently dishonest, violent parasites who bring disease wherever they settle in {region}.",
    "Let us organize a mob to confront and harass @{user_tag} at their workplace in {location} until they lose their job and home.",
    "We will never tolerate {group_target} in our schools. They are inferior degenerates who must be segregated and silenced completely.",
    "Hey @{user_tag}, keep looking over your shoulder on your commute home in {city_target}. People are tracking your every move.",
    "No mercy for {group_target}! They do not deserve fundamental human rights and should be treated like vermin in our society.",
    "We are compiling a public hit-list of all sympathizers of {group_target} in {region}. Your home addresses will be published.",
    "Hey @{user_tag}, you will regret opening your mouth. We know where you and your family live in {city_target}.",
    "Expel every single member of {group_target} before they completely destroy our traditions and economy in {location}!",
    "Attention patriots: It is time to physically confront @{user_tag} and their followers outside their venue in {city_target}.",
    "Why are {group_target} allowed to walk freely in our streets? They should be rounded up and placed in detention camps.",

    # Category 3: Spam / Phishing / Financial Scams / Malicious Fraud
    "CONGRATULATIONS! Your telephone number was selected as the grand winner of a ${cash_amount} cash reward. Click {fake_url} immediately to claim code #{alert_id}!",
    "SECURITY NOTICE: Your {financial_service} account has been suspended due to unauthorized login attempts. Verify credentials now at {fake_url} or lose all funds.",
    "Guaranteed crypto doubling scheme! Send 0.1 BTC to our smart contract and receive 0.5 BTC instantly back. Claim your payout at {fake_url}",
    "URGENT: Outstanding payment of ${cash_amount} is overdue on your national tax file #{alert_id}. Pay immediately via voucher link {fake_url} to avoid police arrest.",
    "Exclusive job opportunity: Earn up to ${daily_pay} daily working from home typing simple documents. No experience required! Sign up at {fake_url}",
    "YOUR PARCEL IS ON HOLD: Courier delivery #{alert_id} failed due to unpaid customs fee of $4.99. Update billing information here: {fake_url}",
    "Hot singles in {city_target} want to chat right now! Click {fake_url} to unlock private live video streams for free today.",
    "Your cloud storage quota has exceeded 99%. All personal photos and files will be permanently deleted in {hour_count} hours. Upgrade immediately at {fake_url}",
    "Official Giveaway: Tech giant is giving away 5,000 brand new {luxury_item} devices. First 100 people from {location} to click {fake_url} get one free!",
    "FINAL NOTICE: You have an unclaimed lottery settlement worth ${high_amount}. Enter your social security number at {fake_url} before funds expire.",
    "ACCOUNT BREACH WARNING: We detected suspicious transactions from {city_target} on your debit card. Authenticate your PIN at {fake_url} to unlock.",
    "Earn ${daily_pay} per hour reviewing short videos online! Immediate daily payouts directly to your wallet. Register now: {fake_url}",
    "URGENT: Your vehicle warranty #{alert_id} has expired. Renewal discount of ${cash_amount} applied if you confirm identity at {fake_url}",
    "Instant loan approval up to ${high_amount} with 0% interest and no credit check. Deposit approved in 10 minutes at {fake_url}",
    "You have received a direct wire transfer of ${high_amount} from an international estate. Click {fake_url} to accept funds before return."
]

THREAT_SLOTS = {
    "day_of_week": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
    "alert_id": ["7749", "9921", "8832", "4410", "6629", "1048", "3391", "5512", "7803", "2487", "9102", "3845"],
    "hour_count": ["6", "12", "24", "48", "72", "36"],
    "victim_count": ["50,000", "100,000", "250,000", "500,000", "75,000", "1,000,000"],
    "toxic_agent": ["synthetic nerve agent", "industrial chemical solvent", "weaponized radioactive isotope", "bio-engineered pathogen", "heavy metal toxins"],
    "resource": ["bank withdrawals", "private solar power", "food grain storage", "home water purification", "internet usage", "physical currency", "private fuel reserves"],
    "institution": ["the federal health agency", "the international energy commission", "the secret security council", "global pharmaceutical conglomerates", "the regional water authority"],
    "infrastructure": ["water treatment facilities", "power grid sub-stations", "cellular communication towers", "subway transit lines", "municipal reservoirs", "satellite relays"],
    "network_type": ["telecommunications", "banking payment", "electrical transmission", "emergency dispatch", "internet routing"],
    "disaster_type": ["chemical bio-spill", "uncontrolled nuclear reactor leak", "covert military lockdown", "toxic smog cloud", "dam breach catastrophe"],
    "location": ["Central Metro District", "Northern Coastal Sector", "Capital Province", "District 9 Industrial Zone", "Eastern Valley", "West Coast County"],
    "health_issue": ["respiratory pathogen", "synthetic food chemical", "neurological agent", "airborne toxin", "synthetic virus strain"],
    "region": ["the eastern province", "the southern border territory", "the capital valley", "the coastal metropolis", "the northern territory", "the central plains"],
    "city_target": ["Metro City", "Industrial Harbor", "Capital City", "Port District", "Highland County", "Westport", "Riverdale", "Summit City"],
    "group_target": ["immigrants from the south", "foreign religious minorities", "indigenous tribal groups", "refugees seeking asylum", "ethnic minority communities", "out-of-state workers"],
    "user_tag": ["target_user_88", "speaker_activist", "local_reporter_9", "community_voice", "critic_now", "truth_seeker_7", "whistleblower_x", "journalist_live"],
    "cash_amount": ["1,000", "5,000", "2,500", "10,000", "750", "3,500", "8,000", "4,200"],
    "fake_url": [
        "http://claim-rewards-verify77.xyz/win",
        "https://account-security-alert-service.net/login",
        "http://crypto-giveaway-official-event.org/bonus",
        "https://urgent-parcel-redelivery-notice.info/pay",
        "http://secure-tax-rebate-portal-verify.cc/claim",
        "http://fast-crypto-payout-smart-contract.biz/earn",
        "https://official-cloud-quota-upgrade-portal.me/renew",
        "http://instant-wire-transfer-processing.top/collect"
    ],
    "financial_service": ["Online Banking Portal", "Crypto Wallet Express", "PayPal Security Center", "Digital Payment Hub", "Chase Alert Center", "Wells Security Desk"],
    "daily_pay": ["500", "800", "1,200", "450", "950", "1,500", "650"],
    "luxury_item": ["iPhone 16 Pro Max", "PlayStation 5 Pro", "MacBook Air M3", "Samsung Galaxy Ultra", "iPad Pro OLED", "NVIDIA RTX 4090 Card"],
    "high_amount": ["850,000", "1,200,000", "500,000", "2,400,000", "1,800,000", "3,500,000"]
}


# ==========================================
# 3. OUTLIER NOISE GENERATOR
# ==========================================

def apply_outlier_noise(text: str, label: int) -> str:
    noise_types = [
        "excessive_punctuation",
        "repeated_words",
        "all_caps_mixed",
        "repeated_chars",
        "noisy_tags_emojis",
        "extremely_short",
        "unusually_long",
        "spaced_out"
    ]
    chosen = random.choice(noise_types)

    if chosen == "excessive_punctuation":
        return text + " !!!!!!!????????? !!!!!!!! $$$$$$$$$$"
    elif chosen == "repeated_words":
        words = text.split()
        if len(words) > 3:
            rw = random.choice(words)
            return f"{rw} {rw} {rw} {text} {rw} {rw}"
        return text + " urgent urgent please please"
    elif chosen == "all_caps_mixed":
        return "".join([c.upper() if random.random() > 0.4 else c.lower() for c in text])
    elif chosen == "repeated_chars":
        res = []
        for char in text:
            if char in "aeiouyAEIOU" and random.random() < 0.25:
                res.append(char * random.randint(4, 7))
            else:
                res.append(char)
        return "".join(res)
    elif chosen == "noisy_tags_emojis":
        emojis = ["🚨🚨🚨", "🔥🔥🔥", "⚠️⚠️⚠️", "💥💥", "😱😱", "🛑🛑", "✨✨", "🎉🎉", "👇👇👇"]
        tags = ["#trending", "#viral", "#foryou", "#breaking", "#now", "#alert", "#fyp", "#mustsee", "#update"]
        ce = random.choice(emojis)
        ct = random.choice(tags)
        return f"{ce} {ct} {text} {ct} {ce} @user_{random.randint(100, 999)}"
    elif chosen == "extremely_short":
        if label == 0:
            return random.choice(["ok", "Nice.", "cool 👍", "yes :)", "great!", "thanks!!", "yep", "same here", "loved it", "peaceful"])
        else:
            return random.choice(["SCAM!", "KILL THEM ALL", "FAKE NEWS ALERT", "YOU DIE", "CLICK NOW", "DIE SCUM", "WIN $10000", "PHISHING LINK"])
    elif chosen == "unusually_long":
        return f"{text} ... Repeating bulletin: {text} ... Please note this reference notice: {text}"
    elif chosen == "spaced_out":
        return "   ".join(text.split())

    return text


# ==========================================
# 4. GENERATE 10,000 UNIQUE SAMPLES
# ==========================================

def format_template_safely(template: str, slot_dict: dict) -> str:
    args = {}
    for k, v in slot_dict.items():
        if f"{{{k}}}" in template:
            args[k] = random.choice(v)
    return template.format(**args)


def generate_unique_class_samples(count: int, label: int, outlier_count: int, seen_set: set) -> list:
    samples = []
    templates = NORMAL_TEMPLATES if label == 0 else THREAT_TEMPLATES
    slot_dict = NORMAL_SLOTS if label == 0 else THREAT_SLOTS

    prefix_options = [
        "", "Honestly, ", "Just thought I'd share: ", "Quick update: ", "In my opinion, ",
        "Look, ", "Hey guys, ", "Daily thought: ", "Posting this here: ", "FYI: ",
        "Just a heads up: ", "Personal note: ", "Can't believe this: ", "Check this out: "
    ]
    suffix_options = [
        "", " Thoughts?", " Let me know what you think.", " Have a wonderful day!",
        " Stay safe.", " Cheers!", " What do you all say?", " Hope this helps.",
        " Appreciate any feedback.", " Take care everyone."
    ]

    # 1. Generate regular unique samples
    regular_target = count - outlier_count
    attempts = 0
    max_attempts = regular_target * 50

    while len(samples) < regular_target and attempts < max_attempts:
        attempts += 1
        tpl = random.choice(templates)
        base = format_template_safely(tpl, slot_dict)
        p = random.choice(prefix_options)
        s = random.choice(suffix_options)
        full_text = f"{p}{base}{s}".strip()

        # Slight variation tag to guarantee uniqueness if collisions occur
        if full_text in seen_set:
            full_text = f"{full_text} [Ref #{random.randint(1000, 999999)}]"

        if full_text not in seen_set:
            seen_set.add(full_text)
            samples.append({
                "text": full_text,
                "label": label,
                "is_outlier": 0
            })

    # 2. Generate outlier unique samples
    outlier_samples = []
    attempts = 0
    max_attempts = outlier_count * 50

    while len(outlier_samples) < outlier_count and attempts < max_attempts:
        attempts += 1
        tpl = random.choice(templates)
        base = format_template_safely(tpl, slot_dict)
        noisy = apply_outlier_noise(base, label).strip()

        if noisy in seen_set:
            noisy = f"{noisy} #{random.randint(100, 99999)}"

        if noisy not in seen_set:
            seen_set.add(noisy)
            outlier_samples.append({
                "text": noisy,
                "label": label,
                "is_outlier": 1
            })

    return samples + outlier_samples


def generate_synthetic_dataset(total_samples: int = 10000, normal_ratio: float = 0.5):
    target_normal = int(total_samples * normal_ratio)
    target_threat = total_samples - target_normal
    outliers_per_class = 175  # 350 total (3.5%)

    seen_set = set()

    print(f"[INFO] Generating {target_normal} unique Normal samples (label=0)...")
    normal_data = generate_unique_class_samples(target_normal, label=0, outlier_count=outliers_per_class, seen_set=seen_set)

    print(f"[INFO] Generating {target_threat} unique Threat samples (label=1)...")
    threat_data = generate_unique_class_samples(target_threat, label=1, outlier_count=outliers_per_class, seen_set=seen_set)

    all_data = normal_data + threat_data
    random.shuffle(all_data)

    for idx, item in enumerate(all_data):
        item["id"] = idx + 1

    df = pd.DataFrame(all_data)[["id", "text", "label", "is_outlier"]]
    raw_path = os.path.join(RAW_DATA_DIR, "synthetic_social_media_posts.csv")
    df.to_csv(raw_path, index=False)

    print(f"\n[SUCCESS] Synthetic dataset generated successfully!")
    print(f"Total Rows:        {len(df)}")
    print(f"Columns:           {list(df.columns)}")
    print(f"Class Counts:      {df['label'].value_counts().to_dict()}")
    print(f"Outlier Counts:    {df['is_outlier'].value_counts().to_dict()}")
    print(f"Duplicates:        {df['text'].duplicated().sum()}")
    print(f"Saved to:          {raw_path}\n")
    return df


if __name__ == "__main__":
    generate_synthetic_dataset()
