import React, { useState } from "react";
import axios from "axios";

function App() {
  const [file, setFile] = useState(null);
  const [results, setResults] = useState([]);

  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    const formData = new FormData();
    formData.append("file", file);

    try {
      console.log(formData);
      const response = await axios.post(
        "https://miniprojectbackend-2iul.onrender.com/predict",
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      );
      setResults(response.data);
    } catch (error) {
      console.error("Error predicting habitability:", error);
      setResults([]);
    }
  };

  return (
    <div className="App">
      <h1>Exoplanet Habitability Prediction</h1>
      <form onSubmit={handleSubmit}>
        <input type="file" onChange={handleFileChange} accept=".csv" required />
        <button type="submit">Upload and Predict</button>
      </form>
      <div className="results">
        <h2>Prediction Results:</h2>
        <ul>
          {results.map((result, index) => (
            <li key={index}>
              <strong>{result.kepoi_name}</strong>: {result.habitable}
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}

export default App;
