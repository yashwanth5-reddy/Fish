
from flask import Flask, request, render_template, redirect, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
import bcrypt
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__, template_folder="templates")
CORS(app)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.secret_key = 'secret_key'
db = SQLAlchemy(app)

# Load models
with open('fish_disease_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('pca.pkl', 'rb') as f:
    pca = pickle.load(f)

with open('mlb.pkl', 'rb') as f:
    mlb = pickle.load(f)

with open('encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

# User Model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)

    def __init__(self, name, email, password):
        self.name = name
        self.email = email
        self.password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    def check_password(self, password):
        return bcrypt.checkpw(password.encode('utf-8'), self.password.encode('utf-8'))

with app.app_context():
    db.create_all()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        
        new_user = User(name=name, email=email, password=password)
        db.session.add(new_user)
        db.session.commit()
        return redirect('/login')
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        user = User.query.filter_by(email=email).first()
        
        if user and user.check_password(password):
            session['email'] = user.email
            return redirect('/dashboard')
        else:
            return render_template('login.html', error='Invalid user')
    
    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    if 'email' in session:
        user = User.query.filter_by(email=session['email']).first()
        return render_template('xtra.html', user=user)  # Redirect to xtra.html after login
    
    return redirect('/login')

@app.route('/logout')
def logout():
    session.pop('email', None)
    return redirect('/')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        season = data.get('season')
        temperature = float(data.get('temperature'))
        ph = float(data.get('ph'))
        salinity = float(data.get('salinity'))
        oxygen = float(data.get('oxygen'))
        ammonia = float(data.get('ammonia'))

        input_data = [oxygen, salinity, ph, temperature, ammonia]
        season_encoded = encoder.transform([[season]])[0]  
        input_array = np.array([input_data + season_encoded.tolist()])
        input_scaled = scaler.transform(input_array)
        input_pca = pca.transform(input_scaled)

        prediction = model.predict(input_pca)
        predicted_disease = mlb.inverse_transform(prediction)
        disease_names = predicted_disease[0] if predicted_disease else ["No Disease Detected"]

        risk_level = "High" if ammonia > 1.0 or oxygen < 3.0 or temperature > 30 else \
                     "Medium" if ammonia > 0.5 or oxygen < 5.0 or temperature > 28 else "Low"

        preventive_measures = {
            "Aeromoniasis": {"Low": [
                    "Maintain stable water conditions with regular filtration and aeration.",
                    "Periodic water exchange to remove organic waste.",
                    "Use immune-boosting supplements in feed.",
                    "Conduct weekly health checks."
                ],
                "Medium": [
                    "Provide nutrient-rich feed with Vitamin C & probiotics.",
                    "Consider vaccination if available.",
                    "Apply mild disinfectants like salt baths (2-3% concentration)."
                ],
                "High": [
                    "Maintain optimal water temperature (avoid sudden fluctuations).",
                    "Administer antibiotic treatments (Oxytetracycline, Sulfonamides).",
                    "Increase dissolved oxygen levels using aerators.",
                    "Use potassium permanganate or formalin for water disinfection."
                ]},
           "Columnaris": {
                "Low": ["Maintain clean water with regular filtration and water exchange",
                        "Conduct weekly health assessments for early detection.",
                        "Ensure balanced nutrition with disease-preventive additives."],
                "Medium": ["Immunity Boosting:Add garlic extract, beta-glucans, and vitamin C to fish feed.",
                           "Ensure adequate aeration and prevent stagnant water.",
                           "Test pH, temperature, ammonia, and salinity regularly."],
                "High": ["Sterilize nets, tanks, and equipment with chlorine-based disinfectants.",
                        "Increase dissolved oxygen and reduce organic matter in water.",
                        "Use potassium permanganate or formalin to disinfect water.",
                         "Use broad-spectrum antibiotics (e.g., Florfenicol, Oxytetracycline).."]
            },
            "Hypoxia": {
                "Low": ["Ensure proper circulation of water in all pond areas",
                        "Ensure good aeration and minimize organic waste.",
                        "Monitor oxygen levels."],
                "Medium": ["Adjust water flow to increase oxygenation in deeper areas.","Increase aeration."],
                "High": ["Apply controlled algaecide treatments to minimize oxygen-consuming decay.",
                         "Conduct immediate water exchange (but avoid temperature shock)",
                         "Maintain floating plants that produce oxygen without blocking sunlight."]
            },
            "Saprolegniasis": {
                "Low": ["Ensure clean water filtration systems are running efficiently.",
                        "Keep fish stress levels low by ensuring proper tank/pond conditions.",
                        "Ensure clean gravel, sand, or bio-substrate in tanks to prevent fungal growth."],
                "Medium": ["Adjust water flow to increase oxygenation in deeper areas.","Maintain stable pH and temperature levels to prevent fungal growth.",
                           "Treat with antifungal agents."],
                "High": ["Disinfect all equipment and quarantine new fish before introduction.",
                         "Improve water quality by reducing organic debris and decaying matter.",
                         "Apply antifungal treatments (e.g., malachite green, formalin, potassium permanganate)."]
            },
            "Ichthyophthiriasis": {
                "Low": ["Perform weekly fish health assessments for early detection.",
                        "Observe for early symptoms like white cysts and scratching behavior.",
                        "Maintain stable temperature and water quality to avoid parasite outbreaks."],
                "Medium": ["Ensure good aeration and minimize organic waste.","Monitor water temperature, pH, and ammonia regularly.","Use salt treatment."],
                "High": ["Use formalin, copper sulfate, or salt treatments for parasite removal.",
                         "Improve filtration and aeration to keep water clean and oxygenated.",
                         "Raise water temperature gradually to 28-30°C to speed up parasite lifecycle."]
            },
            "No Disease Detected": {
                "Low": ["Continue monitoring."],
                "Medium": ["Adjust water parameters."],
                "High": ["Optimize oxygen levels."]
            }
        }


        measures = [preventive_measures.get(disease, {}).get(risk_level, "General water quality maintenance.")
                    for disease in disease_names]

        return jsonify({'prediction': disease_names, 'risk_level': risk_level, 'preventive_measures': measures})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)

