const onnx = require('onnxjs-node');
const { Tensor } = onnx;

const model = new onnx.InferenceSession();
const ACC_WINDOW_SIZE = 10;

const activityElement = document.getElementById("activity");

// Function to update the activity and background color
function updateActivity(prediction) {
    // Update the text content with the prediction
    activityElement.textContent = prediction;

    // Set default background color
    activityElement.style.backgroundColor = "white";

    // Change background color based on the prediction
    if (prediction === "Walking") {
        activityElement.style.backgroundColor = "blue";
    } else if (prediction === "Standing") {
        activityElement.style.backgroundColor = "white";
    } else if (prediction === "Running") {
        activityElement.style.backgroundColor = "red";
    }
}

// Function to calculate rolling average of an array
function rollingAverage(arr) {
    return arr.reduce((a, b) => a + b, 0) / arr.length;
}

// Function to calculate rolling variance of an array
function rollingVariance(arr, avg) {
    return arr.reduce((a, b) => a + (b - avg) ** 2, 0) / arr.length;
}

async function initialize() {
    try {
        await model.loadModel('/Users/rweis/Desktop/Exercise/onnx/pipeline_xgboost.onnx');
        console.log('Model loaded successfully.');
    } catch (error) {
        console.error('Error loading the ONNX model:', error);
    }

    if (typeof DeviceMotionEvent !== "undefined" && typeof DeviceMotionEvent.requestPermission === "function") {
        try {
            const response = await DeviceMotionEvent.requestPermission();
            console.log('Device motion permission granted.');
        } catch (error) {
            console.error('Error requesting device motion permission:', error);
        }
    }

    let acc_x_window = new Array(ACC_WINDOW_SIZE).fill(0);
    let acc_y_window = new Array(ACC_WINDOW_SIZE).fill(0);
    let acc_z_window = new Array(ACC_WINDOW_SIZE).fill(0);

    window.addEventListener('devicemotion', async (event) => {
        if (event.acceleration) {
            acc_x_window.push(event.acceleration.x);
            acc_x_window.shift();

            acc_y_window.push(event.acceleration.y);
            acc_y_window.shift();

            acc_z_window.push(event.acceleration.z);
            acc_z_window.shift();

            const acc_x_avg = rollingAverage(acc_x_window);
            const acc_y_avg = rollingAverage(acc_y_window);
            const acc_z_avg = rollingAverage(acc_z_window);

            const acc_x_var = rollingVariance(acc_x_window, acc_x_avg);
            const acc_y_var = rollingVariance(acc_y_window, acc_y_avg);
            const acc_z_var = rollingVariance(acc_z_window, acc_z_avg);

            const input = new Tensor(
                new Float32Array([acc_x_avg, acc_y_avg, acc_z_avg, acc_x_var, acc_y_var, acc_z_var]),
                'float32',
                [1, 6]
            );

            // normalize input
            input.data = input.data.map((x, i) => (x - input.mean[i]) / input.std[i]);
            
            // check that model is loaded and can be used for inference
            if (!model || !model.session || !model.session.isRunning) {
                return;
            }

            try {
                const outputMap = await model.run([input]);
                const outputTensor = outputMap.values().next().value;
                const predictions = outputTensor.data;
                console.log('Predictions:', predictions);
                // TODO: Handle predictions as needed
            } catch (error) {
                console.error('Error running inference:', error);
            }
        }
    });
}

initialize();
