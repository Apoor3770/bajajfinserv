import os
import numpy as np
from flask import Flask, request, jsonify, session
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from datetime import timedelta
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///app.db'  # Change to MySQL if needed
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=30)

# Initialize extensions
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

# Define User Model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(128), nullable=False)
    activities = db.relationship('Activity', backref='user', lazy=True)

# Define Activity Model
class Activity(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    activity_type = db.Column(db.String(50), nullable=False)
    count = db.Column(db.Integer, default=0)
    timestamp = db.Column(db.DateTime, server_default=db.func.now())
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

# Create database tables
with app.app_context():
    db.create_all()

# Load Pretrained Model (MobileNetV2 as a Placeholder for Activity Recognition)
model = MobileNetV2(weights='imagenet')

def predict_activity(image_path):
    """
    Dummy function to simulate activity recognition.
    Uses MobileNetV2 for image classification and maps results to activities.
    """
    try:
        image = load_img(image_path, target_size=(224, 224))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)
        preds = model.predict(image)
        decoded = decode_predictions(preds, top=1)[0]
        label = decoded[0][1]
        activity_map = {
            'tennis_ball': 'squat',
            'golf_ball': 'pushup',
            'jersey': 'crunch'
        }
        return activity_map.get(label, "unknown")
    except Exception as e:
        print("Prediction error:", e)
        return "unknown"

# Default Home Route
@app.route('/')
def home():
    return """
    <h1>Welcome to the Activity Recognition App</h1>
    <p>Available endpoints:</p>
    <ul>
        <li>/register - Register a new user (POST)</li>
        <li>/login - Login with your credentials (POST)</li>
        <li>/predict - Upload an image to predict an activity (POST)</li>
        <li>/leaderboard - View the leaderboard (GET)</li>
        <li>/profile - View your profile (GET)</li>
    </ul>
    """

# User Registration Route
@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    if not data or 'username' not in data or 'email' not in data or 'password' not in data:
        return jsonify({'message': 'Invalid request'}), 400

    hashed_password = bcrypt.generate_password_hash(data['password']).decode('utf-8')
    new_user = User(username=data['username'], email=data['email'], password=hashed_password)
    db.session.add(new_user)
    db.session.commit()
    return jsonify({'message': 'User registered successfully'})

# User Login Route
@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    if not data or 'email' not in data or 'password' not in data:
        return jsonify({'message': 'Invalid request'}), 400

    user = User.query.filter_by(email=data['email']).first()
    if user and bcrypt.check_password_hash(user.password, data['password']):
        session['user_id'] = user.id
        return jsonify({'message': 'Login successful'})
    return jsonify({'message': 'Invalid credentials'}), 401

# Activity Prediction Route
@app.route('/predict', methods=['POST'])
def predict():
    if 'user_id' not in session:
        return jsonify({'message': 'Please log in first'}), 401
    if 'image' not in request.files:
        return jsonify({'message': 'No image provided'}), 400

    image_file = request.files['image']
    filename = 'temp.jpg'
    image_file.save(filename)
    activity = predict_activity(filename)
    os.remove(filename)

    if activity != "unknown":
        new_activity = Activity(user_id=session['user_id'], activity_type=activity, count=1)
        db.session.add(new_activity)
        db.session.commit()

    return jsonify({'predicted_activity': activity})

# Leaderboard Route
@app.route('/leaderboard', methods=['GET'])
def leaderboard():
    results = (
        db.session.query(User.username, db.func.sum(Activity.count).label('total'))
        .join(Activity, User.id == Activity.user_id)
        .group_by(User.id)
        .order_by(db.func.sum(Activity.count).desc())
        .all()
    )
    return jsonify([{'username': r[0], 'total_activity': r[1]} for r in results])

# Profile Route
@app.route('/profile', methods=['GET'])
def profile():
    if 'user_id' not in session:
        return jsonify({'message': 'Please log in first'}), 401

    user = User.query.get(session['user_id'])
    return jsonify({'username': user.username, 'email': user.email})

# Run Flask App
if __name__ == '__main__':
    app.run(debug=True)
