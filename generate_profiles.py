import json
import random

niches = ["Fitness", "Food Creator", "Tech", "Fashion", "Travel", "Photography", "Wellness"]
cities_india = [
    ("New Delhi", "Delhi", "Delhi, India", "🏋️‍♂️"),
    ("Mumbai", "Maharashtra", "Mumbai, Maharashtra, India", "🍜"),
    ("Bengaluru", "Karnataka", "Bengaluru, Karnataka, India", "💻"),
    ("Jaipur", "Rajasthan", "Jaipur, Rajasthan, India", "📸"),
    ("Goa", "", "Goa, India", "🌴"),
    ("Hyderabad", "Telangana", "Hyderabad, Telangana, India", "🚀"),
    ("Pune", "Maharashtra", "Pune, Maharashtra, India", "☕")
]

first_names = ["Aarav", "Vivaan", "Aditya", "Vihaan", "Arjun", "Sai", "Reyansh", "Ayaan", "Krishna", "Ishaan", "Diya", "Sanya", "Ananya", "Priya", "Riya", "Neha", "Kavya", "Tanvi"]
last_names = ["Sharma", "Verma", "Gupta", "Malhotra", "Mehra", "Joshi", "Patel", "Reddy", "Iyer", "Choudhary", "Singh", "Kumar", "Das"]

bios = {
    "Fitness": "Transforming lives through science-based fitness, strength training & clean nutrition 💪 Let's achieve your goals together.",
    "Food Creator": "Exploring hidden food gems, street food & creating delicious fusion recipes 🍜 Food blogger sharing daily cravings.",
    "Tech": "Software developer & AI enthusiast 💻 Sharing coding tutorials, tech reviews, and building cool projects.",
    "Fashion": "Sustainable streetwear & ethnic fashion styling 👗 Elevating everyday looks with modern aesthetics.",
    "Travel": "Wanderlust soul exploring incredible landscapes across India & beyond ✈️ Join my travel adventures.",
    "Photography": "Capturing raw street moments and cinematic landscapes through my lens 📸",
    "Wellness": "Yoga practitioner, mental health advocate & mindful living guide 🌿 Find your inner peace."
}

profiles = []
for i in range(1, 10000):
    niche = random.choice(niches)
    city, state, location, emoji = random.choice(cities_india)
    fname = random.choice(first_names)
    lname = random.choice(last_names)
    username = f"{fname.lower()}_{lname.lower()}_{random.randint(10, 999)}"
    
    profile = {
        "id": f"p{i+100}",
        "username": username,
        "name": f"{fname} {lname}",
        "category": niche,
        "location": location,
        "city": city,
        "country": "India",
        "follower_count": random.randint(10500, 550000), # 10k se zyada followers
        "following_count": random.randint(200, 900),
        "posts_count": random.randint(150, 1200),
        "engagement_rate": round(random.uniform(2.5, 7.5), 1),
        "verified": random.choice([True, False]),
        "bio": bios[niche],
        "recent_posts_tags": [niche.lower().replace(" ", ""), city.lower(), "creator", "trending", "india"],
        "external_link": f"https://{username}.com",
        "avatar_emoji": emoji
    }
    profiles.append(profile)

# Save to data/sample_profiles.json
with open("data/sample_profiles.json", "w", encoding="utf-8") as f:
    json.dump(profiles, f, indent=4)

print("500 Indian profiles successfully generated and saved to data/sample_profiles.json!")