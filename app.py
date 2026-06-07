from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# MEMBACA DATASET
df = pd.read_csv("diabetes_raw.csv")

# KLASIFIKASI TARGET DARI KADAR GULA DARAH (glyhb)
# Tidak (<5.7), Terkena (>=5.7)
def label_glyhb(val):
    if val < 5.7:
        return "Tidak"
    else:
        return "Terkena"

df['target'] = df['glyhb'].apply(label_glyhb)

# FITUR (jangan gunakan glyhb sebagai fitur)
features = ['stab.glu', 'weight', 'waist', 'hip']
full = df[features + ['target']].dropna()
X = full[features]
y = full['target']

# TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# SCALING
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# MODEL RANDOM FOREST
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train_scaled, y_train)

# EVALUASI
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label='Terkena')
recall = recall_score(y_test, y_pred, pos_label='Terkena')
f1 = f1_score(y_test, y_pred, pos_label='Terkena')

app = Flask(__name__)

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        if data is None:
            return jsonify({'error': 'Request body harus berupa JSON'}), 400

        required = ['glucose', 'weight', 'waist', 'hip']
        for field in required:
            if field not in data or data[field] is None or data[field] == '':
                return jsonify({'error': f'Field "{field}" wajib diisi'}), 400

            try:
                val = float(data[field])
            except (ValueError, TypeError):
                return jsonify({'error': f'Field "{field}" harus berupa angka'}), 400

            if val <= 0:
                return jsonify({'error': f'Field "{field}" harus lebih dari 0'}), 400

        glucose = float(data['glucose'])
        weight_kg = float(data['weight'])
        weight_lbs = weight_kg * 2.20462
        waist_cm = float(data['waist'])
        waist_in = waist_cm / 2.54
        hip_cm = float(data['hip'])
        hip_in = hip_cm / 2.54

        new_data = pd.DataFrame({
            'stab.glu': [glucose],
            'weight': [weight_lbs],
            'waist': [waist_in],
            'hip': [hip_in]
        })

        new_scaled = scaler.transform(new_data)
        prediction = model.predict(new_scaled)[0]
        proba = model.predict_proba(new_scaled)[0]
        confidence = round(max(proba) * 100, 1)

        return jsonify({
            'classification': prediction,
            'confidence': confidence,
            'accuracy': round(accuracy * 100, 2),
            'precision': round(precision * 100, 2),
            'recall': round(recall * 100, 2),
            'f1_score': round(f1 * 100, 2)
        })

    except Exception as e:
        return jsonify({'error': f'Terjadi kesalahan: {str(e)}'}), 500

if __name__ == "__main__":
    app.run(debug=True)
