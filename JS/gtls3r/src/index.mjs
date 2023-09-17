import edgeML from "edge-ml"

let collector = undefined;
let isRecording = false;

const deviceApiKey = "9c089f329ae8b051241bd321a3f18d89"; // Set your API key here

// Reference to the toggle switch
var toggleSwitch = document.getElementById("toggle-switch");

// Function to start recording
async function startRecording() {
  if (typeof DeviceMotionEvent !== "undefined" && typeof DeviceMotionEvent.requestPermission === "function") {
    // (optional) Do something before API request prompt.
    try {
      const response = await DeviceMotionEvent.requestPermission();
      isRecording = true;
      
      // const collector = require("edge-ml").datasetCollector;
        collector = await edgeML.datasetCollector(
        "https://app.edge-ml.org",
        deviceApiKey,
        "datasetName",
        true,
        ["accX", "accY", "accZ"],
        { key: "value" },
        "acc_acc"
      );
    } catch (e) {
      console.log(e);
    }
  }
}

// Function to stop recording
async function stopRecording() {
  if (isRecording) {
    // Stop recording
    await collector.onComplete();
    isRecording = false;
  }
}

// Add an event listener to the toggle switch
toggleSwitch.addEventListener("change", () => {
  if (toggleSwitch.checked) {
    // Toggle switch is on, start recording
    startRecording();
  } else {
    // Toggle switch is off, stop recording
    stopRecording();
  }
});

window.addEventListener('devicemotion', async (event) => {
  if (isRecording  && event.acceleration) {
    console.log(event.acceleration)
    try {
      if(event.acceleration.x){
        collector.addDataPoint('accX', event.acceleration.x);
      }
      if(event.acceleration.x){
        collector.addDataPoint('accY', event.acceleration.y);
      }
      if(event.acceleration.x){
        collector.addDataPoint('accZ', event.acceleration.z);
      }
    } catch (e) {
      console.error(`Error adding data point: ${e.message}`);
    }
  }
});




