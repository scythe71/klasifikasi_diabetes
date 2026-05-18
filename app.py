# _WEB APP KLASIFIKASI DIABETES_
# _MENGGUNAKAN FLASK + K-MEANS_

# Install library:
# pip install flask pandas matplotlib scikit-learn

# _IMPORT LIBRARY_

from flask import Flask, render_template_string, request
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# MEMBACA DATASET
df = pd.read_csv("diabetes_raw.csv")

# _MEMILIH FITUR_
X = df[['stab.glu', 'weight', 'waist', 'hip']]

# Menghapus data kosong
X = X.dropna()

# Menyesuaikan dataframe
df = df.loc[X.index]

# _NORMALISASI DATA_
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# _K-MEANS CLUSTERING_
# Jumlah cluster = 5
kmeans = KMeans(n_clusters=5, random_state=42)

# Training model
df['Cluster'] = kmeans.fit_predict(X_scaled)

# _MENGURUTKAN CLUSTER_
# _BERDASARKAN GULA DARAH_
cluster_mean = df.groupby('Cluster')['stab.glu'].mean().sort_values()

# Label klasifikasi
labels = [
    "Not infected",
    "Semi infected",
    "Infected (little)",
    "Infected (worse)",
    "Infected (dangerous)"
]

# _MAPPING CLUSTER_
mapping = {}

for i, cluster_id in enumerate(cluster_mean.index):
    mapping[cluster_id] = labels[i]

# Menambahkan klasifikasi
df['Classification'] = df['Cluster'].map(mapping)

# _EVALUASI MODEL_
# Inertia
inertia = kmeans.inertia_

# Silhouette Score
silhouette = silhouette_score(
    X_scaled,
    df['Cluster']
)

# _MEMBUAT FLASK APP_
app = Flask(__name__)

# _HTML TEMPLATE_
HTML = """

<!DOCTYPE html>
<html>

<head>

    <title>Diabetes Classification</title>

    <style>

        body{
            font-family: Arial;
            background:#f2f2f2;
            padding:40px;
        }

        .container{
            background:white;
            padding:30px;
            border-radius:10px;
            width:500px;
            margin:auto;
            box-shadow:0px 0px 10px gray;
        }

        input{
            width:100%;
            padding:10px;
            margin-top:10px;
            margin-bottom:10px;
        }

        button{
            margin-top:20px;
            padding:12px;
            width:100%;
            background:blue;
            color:white;
            border:none;
            cursor:pointer;
            border-radius:5px;
        }

        button:hover{
            background:darkblue;
        }

        h1{
            text-align:center;
        }

        .result{
            margin-top:20px;
            padding:15px;
            background:#e6ffe6;
            border-radius:10px;
        }

    </style>

</head>

<body>

<div class="container">

    <h1>Diabetes Classification</h1>

    <form method="POST">

        <label>Blood Glucose</label>
        <input type="number" name="glucose" required>

        <label>Weight</label>
        <input type="number" name="weight" required>

        <label>Waist</label>
        <input type="number" name="waist" required>

        <label>Hip</label>
        <input type="number" name="hip" required>

        <button type="submit">
            Predict Classification
        </button>

    </form>

    {% if result %}

    <div class="result">

        <h2>Prediction Result</h2>

        <p>
            <b>Classification:</b>
            {{ result }}
        </p>

        <p>
            <b>Inertia:</b>
            {{ inertia }}
        </p>

        <p>
            <b>Silhouette Score:</b>
            {{ silhouette }}
        </p>

    </div>

    {% endif %}

</div>

</body>
</html>

"""

# _ROUTE UTAMA_
@app.route("/", methods=["GET", "POST"])

def home():

    result = None

    if request.method == "POST":

        # Mengambil input user
        glucose = float(request.form['glucose'])
        weight = float(request.form['weight'])
        waist = float(request.form['waist'])
        hip = float(request.form['hip'])

        # Membuat dataframe data baru
        new_data = pd.DataFrame({
            'stab.glu': [glucose],
            'weight': [weight],
            'waist': [waist],
            'hip': [hip]
        })

        # Normalisasi data baru
        new_scaled = scaler.transform(new_data)

        # Prediksi cluster
        predicted_cluster = kmeans.predict(new_scaled)

        # Mengambil hasil klasifikasi
        result = mapping[predicted_cluster[0]]

        # Menambahkan klasifikasi ke data baru
        new_data['Classification'] = result

        # Menyimpan hasil ke CSV
        new_data.to_csv(
            "prediksi_data_baru.csv",
            mode='a',
            header=False,
            index=False
        )

    return render_template_string(
        HTML,
        result=result,
        inertia=round(inertia, 3),
        silhouette=round(silhouette, 3)
    )


# _MENJALANKAN FLASK_
if __name__ == "__main__":

    app.run(debug=True)