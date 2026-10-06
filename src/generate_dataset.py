import os
import sys
import random
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import set_seed, ensure_directories, RAW_DATA_DIR

set_seed(42)
ensure_directories()

# ==============================================================================
# 1. DIVERSE VOCABULARY & SLOTS
# ==============================================================================

USERS = [
    "alex_k", "jordan99", "sam_tech", "maya_dev", "chris_r", "taylor_m",
    "morgan_v", "dev_dan", "sarah_p", "priya_s", "marcus_w", "elena_b",
    "kevin_t", "rachel_g", "david_h", "claire_n", "lucas_m", "zoe_c",
    "hannah_b", "ryan_j", "natalie_k", "ben_d", "olivia_w", "ethan_s",
    "cyber_fan", "tech_lead", "student_22", "daily_coder", "coder_sam",
    "jessica_m", "noah_p", "adam_f", "sophia_l", "liam_t", "chloe_k",
    "jack_v", "emma_d", "daniel_s", "grace_h", "oliver_q", "ava_r"
]

CITIES = [
    "Chicago", "London", "Austin", "Toronto", "Seattle", "Berlin", "Denver",
    "Melbourne", "Boston", "Tokyo", "Singapore", "Dublin", "Atlanta", "Miami",
    "Dallas", "Vancouver", "Portland", "Amsterdam", "San Jose", "San Diego",
    "Philadelphia", "Manchester", "Calgary", "Sydney", "Houston", "Minneapolis",
    "Phoenix", "Edinburgh", "Zurich", "Montreal", "Pittsburgh", "Brisbane",
    "Auckland", "Stockholm", "Copenhagen", "Munich", "Frankfurt", "Vienna"
]

GAMES = [
    "Valorant", "Elden Ring", "Cyberpunk", "Minecraft", "Overwatch", "Apex Legends",
    "Call of Duty", "Fortnite", "League of Legends", "Destiny 2", "Counter-Strike", "Dark Souls",
    "Helldivers 2", "World of Warcraft", "Rocket League", "Dota 2", "Rainbow Six Siege",
    "Baldur's Gate 3", "Final Fantasy XIV", "Street Fighter 6", "Starfield", "Halo Infinite"
]

MOVIES_SHOWS = [
    "The Dark Knight", "Inception", "Stranger Things", "Breaking Bad", "The Last of Us",
    "Succession", "Dune Part Two", "Oppenheimer", "Severance", "The Bear",
    "Shogun", "House of the Dragon", "True Detective", "The Batman", "Interstellar",
    "Blade Runner", "Gladiator", "Matrix", "Fargo", "Mindhunter", "Fight Club"
]

PROGRAMMING_LANGS = [
    "Python", "Rust", "TypeScript", "C++", "Go", "Java", "Kotlin", "Swift", "C#", "JavaScript", "SQL", "Scala"
]

SPORTS_TEAMS = [
    "Arsenal", "Lakers", "Warriors", "Real Madrid", "Celtics", "Chiefs", "Eagles", "Liverpool",
    "Manchester City", "49ers", "Barcelona", "Bayern Munich", "Yankees", "Dodgers", "Heat"
]

FOODS = [
    "homemade sourdough", "ramen noodles", "mushroom risotto", "wood-fired pizza",
    "pasta carbonara", "avocado toast", "tacos al pastor", "lentil soup", "grilled salmon",
    "blueberry pancakes", "chicken curry", "falafel wrap", "margherita pizza", "pad thai",
    "vegetable stir-fry", "bagels with cream cheese", "french toast", "dumplings", "udon soup"
]

COFFEE_TEA = [
    "iced oat milk latte", "pour-over coffee", "matcha latte", "earl grey tea",
    "cappuccino", "cold brew", "green tea", "espresso", "chamomile tea", "peppermint tea", "chai latte",
    "flat white", "black coffee", "iced americano", "jasmine green tea", "hot cocoa"
]

DOMAINS = [
    "secure-verify-auth77.xyz", "portal-update-alert.net", "claim-payouts-portal.org",
    "parcel-redelivery-notice.info", "tax-refund-processing.cc", "instant-crypto-rewards.biz",
    "cloud-quota-upgrade.me", "wire-transfer-confirm.top", "wallet-sync-service.top",
    "customer-banking-auth.info", "express-account-review.xyz", "payout-approval-desk.net"
]

# ==============================================================================
# 2. NORMAL TEMPLATES (BENIGN, HARD NEGATIVES, SLANG & AMBIGUOUS)
# ==============================================================================

NORMAL_TEMPLATES = [
    # Standard Everyday Life / Hobbies / Tech
    "Starting my morning with a warm cup of {drink} and reviewing notes for my {lang} project.",
    "Just finished a {dist}-mile run around {city}. Beautiful clear skies and great weather today!",
    "Finally finished setting up my new desk in the study. Cable management took forever.",
    "Baking {food} on a Sunday afternoon is honestly the best therapy after a long week.",
    "Can anyone recommend a good lightweight mechanical keyboard for coding in {lang}?",
    "Spent the entire afternoon reading a {genre} book at the local library in {city}.",
    "Just wrapped up a productive study session with @{user}. Data structures are clicking now.",
    "Had lunch at this cozy little diner downtown in {city}. Their {food} was incredible.",
    "Listening to a fascinating podcast about astrophysics and deep space exploration.",
    "Planting fresh herbs on the balcony garden today: basil, rosemary, and mint.",
    "Weekend road trip to the mountains near {city} with friends. The scenic view was worth the long drive.",
    "Practicing fingerstyle acoustic guitar for {hours} hours every day is starting to pay off.",
    "Volunteering at the local animal shelter in {city} this morning was so heartwarming.",
    "Nothing beats a hot bowl of {food} after a long cold day at university.",
    "Organized all my research papers and thesis references in Zotero today. Feeling accomplished.",
    "Attended an inspiring virtual tech meetup on open-source software and {lang} tooling.",
    "Watching the sunset by the waterfront in {city}. Nature never fails to amaze me.",
    "Meal prepping lunches for the upcoming work week. Saves so much time and money.",
    "Friendly reminder to stay hydrated, stretch your back, and take screen breaks today.",
    "Finished painting a watercolor landscape of the hills near {city}.",
    "Working from a quiet coffee shop in {city} today with an {drink}. Great ambient noise for coding.",
    "Completed my first {dist}k race this morning! Exhausted but proud of the personal best time.",
    "Cleaned out my email inbox for the first time in months. Zero unread emails feels amazing.",
    "The campus library during finals week is packed, but found a quiet corner on the third floor.",
    "Enjoying a slow Saturday morning reading tech articles and sipping {drink}.",
    "Finally submitted the final draft of our semester project report on {lang} applications.",
    "Visited the art museum in {city} today. The photography exhibition was stunning.",
    "Trying out a new recipe for {food} tonight with fresh ingredients from the farmers market.",
    "Watching old episodes of {movie} with family. The nostalgia is real.",
    "Stargazing from the rooftop tonight. You can see Jupiter and the constellations so clearly.",

    # Hard Negatives (Trigger words like attack, destroy, kill, bomb, hack, virus in benign contexts)
    "We are going to totally destroy the enemy squad in {game} tonight! Squad up @{user}!",
    "That boss fight in {game} killed my character at least {num} times before I beat it.",
    "Our guild executed a flawless attack on the fortress in {game}. Total domination!",
    "I'm going to slaughter everyone in this multiplayer match on {game} lol.",
    "The graphics in {game} are literally killer. Best game of the year hands down.",
    "We need to execute this stealth attack carefully or the entire squad dies in {game}.",
    "Just got an insane {num}-kill streak in {game}! Best game of the week.",
    "That sniper in {game} kept hunting me down across the whole map. Respect the skill.",
    "I am literally dying of laughter at this comedy clip @{user} sent me 😂",
    "This new album by my favorite artist is an absolute bomb! Every track is pure fire 🔥",
    "I am totally dead after that leg workout at the gym in {city} today. Can barely walk.",
    "The spicy salsa at this restaurant is lethal! My mouth is on fire but it tastes amazing.",
    "You are killing it with that presentation today @{user}! Congratulations on the promotion!",
    "That final exam was absolute murder, but I think I managed to pass with decent marks.",
    "My feet are killing me after walking around {city} all day during the street festival.",
    "She completely destroyed the competition during the university debate tournament.",
    "This afternoon caffeine crash is brutal, need an {drink} to power through.",
    "PSA: Be careful of phishing emails pretending to be bank alerts. Never click suspicious links.",
    "Security tip: Enable two-factor authentication on your accounts to prevent unauthorized access.",
    "Our university IT department sent a warning about malware attachments in spam emails.",
    "Installed a new firewall rule today to block automated brute-force attacks on our test server.",
    "Interesting security conference talk on how researchers identify malware and zero-day exploits.",
    "Make sure to update your operating system to patch the recent remote execution vulnerability.",
    "Our team performed a simulated penetration test to identify weak passwords in the system.",
    "Doctors in {city} are warning that the seasonal flu virus is spreading rapidly. Wash your hands!",
    "Had a terrible headache all morning, but a cup of {drink} and some rest helped a lot.",
    "Biomedical researchers published a great paper on neutralizing airborne viruses.",
    "Seasonal allergies are attacking my sinuses today. Can't stop sneezing.",
    "The {team} defense completely crushed their opponents in the championship yesterday!",
    "Rewatching {movie} tonight. The villain's monologue at the end gives me chills.",
    "The fight choreography in {movie} was insane. Loved the action sequences.",
    "That plot twist in {movie} completely blew my mind. Did not see the betrayal coming.",
    "The weather forecast is calling for a severe winter storm warning near {city} this weekend.",
    "Be on the lookout for fake product reviews when shopping online during sales.",

    # Heated Debates, Frustrations & Everyday Complaints
    "I completely disagree with your take on {lang}. The memory safety model is vastly superior.",
    "Honestly, the referee in the {team} match made some terrible calls today. Completely ruined the game.",
    "I hate waking up at 5 AM for early shifts, but {drink} makes it somewhat tolerable.",
    "People who don't return shopping carts to the corral in {city} are the absolute worst.",
    "Why is traffic in {city} always a disaster on Friday afternoons? Takes {hours} hours to get home.",
    "This software update broke half my configuration files. So frustrating to debug!",
    "I don't understand the hype around {movie}. The plot was mediocre and the ending felt rushed.",
    "Can we stop with pointless corporate meetings that could have been a single email?",
    "Public transit delays in {city} are getting out of hand lately. We need better infrastructure.",
    "Hot take: {drink} is overrated. Nothing beats a simple glass of cold water.",
    "The price of groceries in {city} has doubled in two years. It's getting ridiculous.",
    "Trying to cancel a gym membership feels like an impossible quest. Why make it so hard?",
    "Airport security lines at {city} airport were chaotic today, barely made my flight connection.",
    "My neighbor decided to mow the lawn at 6 AM on a Saturday. Absolutely thrilled.",
    "Ordered {food} delivery and they forgot the dipping sauce. Evening ruined."
]

# ==============================================================================
# 3. THREAT TEMPLATES (OVERT, SUBTLE, PHISHING, DISINFORMATION)
# ==============================================================================

THREAT_TEMPLATES = [
    # Overt Harassment, Hate Speech & Violence
    "People like you @{user} are a disease in {city}. You need to be forcefully removed from our society.",
    "We are going to find out where @{user} lives in {city} and teach you a painful lesson in person.",
    "All members of that ethnic minority are vermin and parasites. They don't deserve any human rights.",
    "Hey @{user}, watch your back when you leave your workplace in {city} tonight.",
    "We should organize a mob to attack their community center in {city} and burn it to the ground.",
    "Nobody wants people like you in this country. Go back to where you came from before something happens.",
    "You are a disgusting subhuman @{user}. I hope you suffer and lose everything you care about.",
    "We have compiled a public list of targets in {city}. You will pay for speaking against our group.",
    "No mercy for traitors. It's time to take direct physical action against @{user} and their allies.",
    "If I ever see your face in {city}, I will personally make sure you never walk again.",
    "Get out of our neighborhood in {city} before local groups take matters into their own hands.",
    "Someone needs to physically silence @{user} once and for all. You have been officially warned.",
    "We will hunt down every single person associated with @{user} and ruin your personal lives.",
    "We know where your children go to school in {city} @{user}. Keep your mouth shut.",
    "Filthy parasites like @{user} deserve to be eliminated from our public spaces completely.",

    # Subtle Threats, Intimidation, Doxxing & Social Engineering
    "It would be a real shame if your employer in {city} found out about your private personal records @{user}.",
    "We know which train you take home from {city} every evening at {time}. You aren't as anonymous as you think.",
    "Keep talking like that online @{user}. Unfortunate accidents happen all the time to people like you.",
    "Just a friendly reminder that we have your family's home address in {city} on file. Tread carefully.",
    "Certain people living in the north district of {city} should remember that glass windows break very easily.",
    "Your home address and phone numbers are circulating on private Telegram channels right now @{user}.",
    "You might want to check your car brakes before driving to work in {city} tomorrow morning @{user}.",
    "People who oppose our organization in {city} have a strange habit of disappearing from public life.",
    "We are keeping close tabs on your daily routine in {city} @{user}. Don't test our patience.",
    "You should be very careful what you post about us if you care about your personal safety in {city}.",
    "Nice house you have in {city} @{user}. Would be awful if something unexpected happened to it tonight.",
    "We have people watching your street in {city}. Best to delete your account and stay quiet from now on.",
    "Hope you have good medical insurance @{user}. You never know who you might bump into on your walk home.",
    "Consider this a private courtesy notice @{user}: one more comment and your full dossier goes public.",
    "You made a dangerous enemy today @{user}. Don't expect to sleep peacefully in {city}.",

    # Phishing, Financial Fraud & Scams
    "URGENT: Your bank account #{num} has been locked due to suspicious logins from {city}. Verify identity at https://{domain}/login now.",
    "Congratulations! You were selected as the winner of a ${cash} cash grant. Click https://{domain}/win to claim code #{num} immediately.",
    "FINAL NOTICE: Customs tax of $4.99 is unpaid on your parcel #{num}. Update payment details at https://{domain}/pay to avoid return.",
    "Guaranteed 10x crypto returns in 24 hours! Send 0.1 ETH to our verified smart contract via https://{domain}/earn",
    "SECURITY ALERT: Unauthorized login attempt detected from {city}. Confirm your credentials at https://{domain}/verify before suspension.",
    "You have an unclaimed settlement wire of ${cash} pending release. Authorize transfer at https://{domain}/transfer before deadline.",
    "Work from home typing simple data and earn ${pay} daily! Zero experience required. Register now at https://{domain}/jobs",
    "Your cloud photo storage is 99% full and will be permanently deleted in {hours} hours. Renew quota here: https://{domain}/renew",
    "Official corporate giveaway: First {num} people from {city} to visit https://{domain}/gift get a free flagship device today!",
    "IRS notification: Overdue tax balance of ${cash} requires immediate resolution. Settle online at https://{domain}/tax to prevent arrest.",
    "Debit card suspended due to unusual foreign transactions in {city}. Reactivate your PIN now at https://{domain}/card",
    "Special promotional bonus: Double your crypto deposit instantly today only at https://{domain}/deposit",
    "Your cellular service will be disconnected in 2 hours due to billing error. Update payment method at https://{domain}/billing",

    # Malicious Disinformation & Fabricated Emergencies
    "EMERGENCY BROADCAST: Water pipelines in {city} have been contaminated with toxic chemical agents. Do not drink municipal tap water!",
    "LEAKED DOCUMENT: Military commanders are preparing mandatory curfews and civilian food confiscations in {city} starting tonight.",
    "Classified hospital report reveals local water supplies were deliberately poisoned with toxic nerve agents in {city}.",
    "The government is secretly shutting down all cellular and electrical power grids in {city} within {hours} hours. Forward this alert!",
    "Whistleblower proof: Commercial banks across {city} will freeze all private ATM cash withdrawals starting tomorrow morning.",
    "Covert biological agent released in subway transit stations across {city}. Mainstream news outlets are actively covering it up!",
    "Secret municipal memo leaked: All public transit in {city} will be suspended indefinitely to control civilian movement.",
    "Emergency order passed in secret: Private vehicles will be impounded across {city} starting Monday. Stock up on supplies now!",
    "Satellite imagery confirms foreign military aircraft entering civilian airspace over {city}. Prepare emergency shelters!"
]

# ==============================================================================
# 4. OUTLIER NOISE GENERATION
# ==============================================================================

def add_noise(text: str) -> str:
    choice = random.choice(["all_caps", "excessive_punct", "repeated_char", "emoji_heavy", "typo", "spaced"])
    if choice == "all_caps":
        return text.upper()
    elif choice == "excessive_punct":
        return text + " !!!???!!! $$$"
    elif choice == "repeated_char":
        words = text.split()
        if words:
            idx = random.randint(0, len(words) - 1)
            words[idx] = words[idx] + "!!!"
            return " ".join(words)
        return text
    elif choice == "emoji_heavy":
        emojis = ["🚨🚨🚨", "⚠️⚠️", "🔥🔥", "👀👀", "💯💯", "✨✨", "‼️‼️", "😱😱", "💥💥"]
        return f"{random.choice(emojis)} {text} {random.choice(emojis)}"
    elif choice == "typo":
        if len(text) > 10:
            pos = random.randint(3, len(text) - 3)
            return text[:pos] + text[pos+1:]
        return text
    elif choice == "spaced":
        return text.replace(" ", "  ")
    return text

# ==============================================================================
# 5. DATASET GENERATION WITH STRICT UNIQUENESS GUARANTEE (10,000 UNIQUE ROWS)
# ==============================================================================

def format_text(template, is_threat, is_outlier):
    text = template.format(
        user=random.choice(USERS),
        city=random.choice(CITIES),
        game=random.choice(GAMES),
        movie=random.choice(MOVIES_SHOWS),
        lang=random.choice(PROGRAMMING_LANGS),
        team=random.choice(SPORTS_TEAMS),
        food=random.choice(FOODS),
        drink=random.choice(COFFEE_TEA),
        genre=random.choice(["sci-fi", "mystery", "history", "philosophy", "biography", "fantasy"]),
        domain=random.choice(DOMAINS),
        dist=random.choice(["3", "5", "8", "10", "12", "15"]),
        hours=random.choice(["2", "3", "4", "6", "12", "24", "48"]),
        num=random.choice(["5", "10", "20", "50", "100", "7749", "8821", "9042", "3319"]),
        cash=random.choice(["2,500", "5,000", "10,000", "15,000", "25,000", "50,000"]),
        pay=random.choice(["450", "600", "750", "900", "1,200", "1,500"]),
        time=random.choice(["5:30 PM", "6:00 PM", "6:45 PM", "7:15 PM", "8:00 PM", "11:00 PM"])
    )
    
    # Random realistic prefixes & suffixes
    prefixes_normal = [
        "", "Just thinking: ", "Honestly, ", "Update: ", "Quick note: ",
        "Good morning! ", "Hey all, ", "So glad that ", "Random thought: ", "PSA: ",
        "Daily log: ", "Personal note: ", "Sharing this: ", "By the way, ", "Weekend update: "
    ]
    suffixes_normal = [
        "", " #daily", " #life", " #tech", " #weekend", " #thoughts", " #study",
        " Hope everyone has a great day!", " Truly enjoying this.", " Loving the process."
    ]
    
    prefixes_threat = [
        "", "ALERT: ", "WARNING: ", "Listen carefully: ", "Attention: ",
        "To everyone in the area: ", "Notice: ", "Urgent update: ", "Heads up: ",
        "Public notice: ", "Important broadcast: ", "Final warning: "
    ]
    suffixes_threat = [
        "", " #alert", " #warning", " #breaking", " #urgent", " #share",
        " You will regret this.", " Don't say you weren't told.", " Act now before it's too late."
    ]
    
    if random.random() < 0.40:
        p = random.choice(prefixes_threat if is_threat else prefixes_normal)
        text = p + text
        
    if random.random() < 0.35:
        s = random.choice(suffixes_threat if is_threat else suffixes_normal)
        text = text + s
        
    if is_outlier:
        text = add_noise(text)
        
    return text


def build_dataset(total_samples=10000, outlier_ratio=0.04):
    print(f"[INFO] Generating exactly {total_samples} UNIQUE samples (5,000 Normal, 5,000 Threat)...")
    
    seen_texts = set()
    normal_samples = []
    threat_samples = []
    
    target_per_class = total_samples // 2
    
    # Generate 5,000 Unique Normal Samples
    while len(normal_samples) < target_per_class:
        is_outlier = (random.random() < outlier_ratio)
        tmpl = random.choice(NORMAL_TEMPLATES)
        text = format_text(tmpl, is_threat=False, is_outlier=is_outlier)
        if text not in seen_texts:
            seen_texts.add(text)
            normal_samples.append({
                "text": text,
                "label": 0,
                "category": "Normal",
                "is_outlier": 1 if is_outlier else 0
            })
            
    # Generate 5,000 Unique Threat Samples
    while len(threat_samples) < target_per_class:
        is_outlier = (random.random() < outlier_ratio)
        tmpl = random.choice(THREAT_TEMPLATES)
        text = format_text(tmpl, is_threat=True, is_outlier=is_outlier)
        if text not in seen_texts:
            seen_texts.add(text)
            threat_samples.append({
                "text": text,
                "label": 1,
                "category": "Threat",
                "is_outlier": 1 if is_outlier else 0
            })
            
    all_samples = normal_samples + threat_samples
    random.shuffle(all_samples)
    
    df = pd.DataFrame(all_samples)
    df["id"] = range(1, len(df) + 1)
    
    out_file = os.path.join(RAW_DATA_DIR, "collected_social_media_posts.csv")
    df.to_csv(out_file, index=False)
    print(f"[INFO] Successfully created {len(df)} UNIQUE samples at {out_file}")
    print(f"       Normal: {(df['label'] == 0).sum()} | Threat: {(df['label'] == 1).sum()} | Outliers: {(df['is_outlier'] == 1).sum()}")
    return df

if __name__ == "__main__":
    build_dataset()
