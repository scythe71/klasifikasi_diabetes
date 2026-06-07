document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('predict-form');
    const resultBox = document.getElementById('result');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        const data = {
            glucose: document.getElementById('glucose').value,
            weight: document.getElementById('weight').value,
            waist: document.getElementById('waist').value,
            hip: document.getElementById('hip').value
        };

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            });

            const result = await response.json();

            if (!response.ok) {
                throw new Error(result.error || 'Prediction failed');
            }

            resultBox.innerText = `Klasifikasi: ${result.classification}`;

            Swal.fire({
                title: 'Hasil Klasifikasi',
                html: `
                    <p><strong>Klasifikasi:</strong> ${result.classification}</p>
                    <p><strong>Confidence:</strong> ${result.confidence}%</p>
                    <hr style="margin:10px 0">
                    <p style="font-size:13px; color:#9ca3af;">Performa Model (kelas Terkena)</p>
                    <p><strong>Akurasi:</strong> ${result.accuracy}%</p>
                    <p><strong>Presisi:</strong> ${result.precision}%</p>
                    <p><strong>Recall:</strong> ${result.recall}%</p>
                    <p><strong>F1-Score:</strong> ${result.f1_score}%</p>
                `,
                icon: 'info'
            });
        } catch (error) {
            resultBox.innerText = 'Klasifikasi gagal. Periksa input Anda.';

            Swal.fire({
                title: 'Error',
                text: error.message,
                icon: 'error'
            });
        }
    });
});
