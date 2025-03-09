from flask import Flask, render_template, request, redirect, url_for, jsonify
from flask_sqlalchemy import SQLAlchemy
from google import genai
from datetime import datetime, timedelta
from collections import defaultdict
from sqlalchemy.exc import IntegrityError
from sqlalchemy import func
import time

# Initialize the Flask app
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
db = SQLAlchemy(app)

# Initialize emotion counters globally
emotion_counters = {
    "angry": 0,
    "sad": 0,
    "stressed": 0,
    "neutral": 0,
    "calm": 0,
    "focused": 0,
    "happy": 0,
    "relaxed": 0,
    "joyous": 0
}

# Define the MoodTracker model
class MoodTracker(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    predicted_mood = db.Column(db.String(20), nullable=False)

    # Mood categories
    angry = db.Column(db.Integer, default=0)
    sad = db.Column(db.Integer, default=0)
    stressed = db.Column(db.Integer, default=0)
    neutral = db.Column(db.Integer, default=0)
    calm = db.Column(db.Integer, default=0)
    focused = db.Column(db.Integer, default=0)
    happy = db.Column(db.Integer, default=0)
    relaxed = db.Column(db.Integer, default=0)
    joyous = db.Column(db.Integer, default=0)

# Define the DailyMoodScore model
class DailyMoodScore(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.Date, nullable=False)
    mood_score = db.Column(db.Float, nullable=False)

    def __repr__(self):
        return f"<DailyMoodScore {self.date} - {self.mood_score}>"

# Track emotions from the recent entries
def track_emotions():
    # Query entries from the last minute
    one_minute_ago = datetime.utcnow() - timedelta(minutes=1)
    recent_entries = MoodTracker.query.filter(MoodTracker.timestamp >= one_minute_ago).all()

    # Reset counters
    local_counters = {emotion: 0 for emotion in emotion_counters}

    # Check all entries for the last minute and increment counters
    for entry in recent_entries:
        for emotion in local_counters:
            if getattr(entry, emotion) > 0:
                local_counters[emotion] += 1

    # Update the global emotion counters
    for emotion, count in local_counters.items():
        emotion_counters[emotion] += count

# Store the daily mood score in the database
def store_daily_mood_score(date, mood_score):
    new_mood_score = DailyMoodScore(date=date, mood_score=mood_score)
    try:
        db.session.add(new_mood_score)
        db.session.commit()
    except IntegrityError:
        db.session.rollback()
        print(f"Error: Mood score for {date} already exists.")

# Add a mood entry in the MoodTracker table
def add_mood_entry(predicted_mood, angry, sad, stressed, neutral, calm, focused, happy, relaxed, joyous):
    new_entry = MoodTracker(
        predicted_mood=predicted_mood,
        angry=angry,
        sad=sad,
        stressed=stressed,
        neutral=neutral,
        calm=calm,
        focused=focused,
        happy=happy,
        relaxed=relaxed,
        joyous=joyous
    )
    db.session.add(new_entry)
    db.session.commit()

# Google API setup
API_KEY = "AIzaSyA0nfoCTrSPi0kczInG7RevdOsZILFpXwY"  # Replace with your actual API key
client = genai.Client(api_key=API_KEY)

# Function to get stress recommendations
def get_stress_recommendations():
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents="Provide 3 single-line recommendations to decrease stress levels."
    )

    if hasattr(response, "text"):
        return response.text.strip().split("\n")
    return ["No response received."]

# Function to generate mood-related text
def get_mood_text(mood):
    prompt = f"""
    The user is feeling "{mood}". Provide a short response based on the mood:
    - If the mood is negative, suggest a single effective remedy.
    - If the mood is positive, provide an uplifting affirmation.
    - Keep responses concise, engaging, and helpful.
    """
    response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
    if hasattr(response, "text"):
        return response.text.strip()
    return "Unable to generate mood insights at this moment."

# Function to fetch the latest predicted mood
def get_latest_predicted_mood():
    latest_entry = MoodTracker.query.order_by(MoodTracker.timestamp.desc()).first()
    return latest_entry.predicted_mood if latest_entry else "unknown"

# Function to map mood to emoji
def get_mood_emoji(mood):
    mood_emoji_map = {
        "angry": "😡",
        "sad": "😢",
        "stressed": "😰",
        "neutral": "😐",
        "calm": "😌",
        "focused": "🎯",
        "happy": "😃",
        "relaxed": "🌿",
        "joyous": "🥳"
    }
    return mood_emoji_map.get(mood, "❓")

# Function to guess the stress level based on mood
def guess_stress_level(mood):
    stress_map = {
        "angry": 90,
        "sad": 70,
        "stressed": 100,
        "neutral": 50,
        "calm": 30,
        "focused": 40,
        "happy": 20,
        "relaxed": 10,
        "joyous": 5
    }
    return stress_map.get(mood, 50)

# Routes for various pages
@app.route("/")
@app.route("/home")
def home():
    mood = get_latest_predicted_mood()
    emoji = get_mood_emoji(mood)
    mood_text = get_mood_text(mood)
    return render_template("Home.html", emoji=emoji, mood_text=mood_text)

@app.route('/warning')
def warning():
    mood = get_latest_predicted_mood()
    current_stress_level = guess_stress_level(mood)
    healthy_stress_level = 50
    
    if current_stress_level > healthy_stress_level:
        recommendations = get_stress_recommendations()
        return render_template('Warning.html', steps=recommendations)
    
    return redirect(url_for('home'))

@app.route("/chatdoctor")
def chatdoctor():
    return render_template("chat.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "")
    
    if not user_message:
        return jsonify({"response": "Please enter a message."})

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=user_message
    )
    bot_response = response.text if response else "Sorry, I couldn't understand that."
    return jsonify({"response": bot_response})

@app.route("/stats")
def stats():
    return render_template("Stats.html")

# Mood score map for daily score calculation
mood_score_map = {
    "angry": 0,
    "sad": 10,
    "stressed": 20,
    "neutral": 50,
    "calm": 60,
    "focused": 70,
    "happy": 80,
    "relaxed": 90,
    "joyous": 100
}

# Calculate the unified daily mood score
def get_daily_mood_score():
    today = datetime.today().date()
    daily_entries = MoodTracker.query.filter(func.date(MoodTracker.timestamp) == today).all()

    if not daily_entries:
        return {"message": "No data for today"}

    mood_counts = defaultdict(int)
    for entry in daily_entries:
        mood_counts[entry.predicted_mood] += 1

    total_moods = len(daily_entries)
    weighted_score = sum(mood_score_map.get(mood, 50) * (count / total_moods) for mood, count in mood_counts.items())
    
    # Normalize and round the score
    normalized_score = weighted_score / 100

    # Return as a dictionary to ensure compatibility with store_daily_mood_score()
    return {"date": today, "unified_mood_score": round(normalized_score, 2)}

@app.route("/daily-mood-score")
def daily_mood_score():
    moodscore = get_daily_mood_score()
    
    # Ensure moodscore contains the 'date' and 'unified_mood_score' keys
    if "date" in moodscore and "unified_mood_score" in moodscore:
        store_daily_mood_score(moodscore["date"], moodscore["unified_mood_score"])
        return render_template("daily-mood-score.html", moodscore=moodscore)
    else:
        # If no data, provide an error or fallback
        return render_template("daily-mood-score.html", moodscore={"message": "No mood data available"})
    
@app.route('/activity', methods=['GET'])
def activity():
    # Track the emotions (update counters)
    track_emotions()
    
    # Return the emotion counters to the template
    return render_template("Activity.html", emotion_counters=emotion_counters)


if __name__ == "__main__":
    app.run(debug=True)
