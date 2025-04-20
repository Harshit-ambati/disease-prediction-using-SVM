const GEMINI_API_KEY = "AIzaSyAT5ZWZZ6TEF7PLGLoAl0y0BtqAciYI8X4"; // Replace with your actual Gemini API key

async function predictDisease() {
  const symptoms = document.getElementById("symptomsInput").value;
  const resultDiv = document.getElementById("result");

  resultDiv.innerHTML = "Analyzing with Gemini AI...";

  const prompt = `
You are a medical assistant. Based on the following symptoms: "${symptoms}", provide:
1. Possible disease name(s)
2. Description of the disease(s)
3. Severity level
4. Common complications
5. Treatments and home remedies
6. Preventive tips
7. Emergency warning signs
`;

  try {
    const response = await fetch("https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=" + GEMINI_API_KEY, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        contents: [{
          parts: [{ text: prompt }]
        }]
      })
    });

    const data = await response.json();
    const reply = data.candidates?.[0]?.content?.parts?.[0]?.text;

    resultDiv.innerText = reply || "Could not generate insights. Try again.";
  } catch (err) {
    console.error(err);
    resultDiv.innerText = "Error contacting Gemini API.";
  }
}
