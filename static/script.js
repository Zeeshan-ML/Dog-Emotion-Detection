// File: static/script.js
const form = document.getElementById('uploadForm');
const resultDiv = document.getElementById('result');
const fileInput = document.getElementById('audioFile');
const fileNameDisplay = document.getElementById('fileName');

fileInput.addEventListener('change', () => {
  if (fileInput.files.length > 0) {
    fileNameDisplay.textContent = `Selected file: ${fileInput.files[0].name}`;
    fileNameDisplay.style.color = '#5566ff';
  } else {
    fileNameDisplay.textContent = 'No file chosen';
    fileNameDisplay.style.color = '#334e68';
  }
});

form.addEventListener('submit', async (e) => {
  e.preventDefault();

  const fileInput = document.getElementById('audioFile');
  if (!fileInput.files.length) {
    alert('Please select an audio file');
    return;
  }

  const file = fileInput.files[0];
  const formData = new FormData();
  formData.append('file', file);

  resultDiv.textContent = "Processing...";

  try {
    const response = await fetch('/predict/', {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      const err = await response.json();
      resultDiv.textContent = `Error: ${err.detail || 'Unknown error'}`;
      return;
    }

    const data = await response.json();
    resultDiv.textContent = `Prediction: ${data.emotion}`;

  } catch (error) {
    resultDiv.textContent = `Error: ${error.message}`;
  }
});
