from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import os

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ===============================
# LOAD MODELS
# ===============================
efficientnet_model = tf.keras.models.load_model("models/efficientnet_model.keras")
cnn_model = tf.keras.models.load_model("models/cnn_model.keras")
resnet_model = tf.keras.models.load_model("models/resnet50_model.keras")

# ===============================
# LABEL MAP (SAME AS GRADIO)
# ===============================
label_map = {
    "0": "NonDemented",
    "1": "VeryMildDemented",
    "2": "MildDemented",
    "3": "ModerateDemented"
}

# ===============================
# GRAD-CAM (SAME AS GRADIO)
# ===============================
def make_gradcam_heatmap(img_array, model, model_name):
    try:
        if model_name == "efficientnet":
            last_conv_layer_name = "top_conv"
        elif model_name == "resnet":
            last_conv_layer_name = "conv5_block3_out"
        elif model_name == "cnn":
            last_conv_layer_name = "conv2d_2"
        else:
            return None

        grad_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=[
                model.get_layer(last_conv_layer_name).output,
                model.output
            ]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            pred_index = int(tf.argmax(predictions[0]))
            loss = predictions[:, pred_index]

        grads = tape.gradient(loss, conv_outputs)

        if grads is None:
            return None

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]

        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        heatmap = tf.maximum(heatmap, 0)
        max_val = tf.reduce_max(heatmap)

        if max_val == 0:
            return None

        heatmap /= max_val
        return heatmap.numpy()

    except Exception as e:
        print("GradCAM Error:", e)
        return None


def overlay_heatmap(heatmap, image):
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    image = np.array(image.resize((224, 224)))
    return cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)

# ===============================
# ROUTES
# ===============================
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files["image"]
        model_choice = request.form["model"]

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # ORIGINAL IMAGE
        original_image = Image.open(filepath).convert("RGB").resize((224, 224))
        img_array_raw = np.array(original_image)
        img_array_raw = np.expand_dims(img_array_raw, axis=0)

        # COPY FOR MODEL
        img_array = img_array_raw.copy()

        # ===============================
        # MODEL-SPECIFIC PREPROCESSING
        # ===============================
        if model_choice == "efficientnet":
            model = efficientnet_model
            img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)

        elif model_choice == "cnn":
            model = cnn_model
            img_array = img_array / 255.0

        elif model_choice == "resnet":
            model = resnet_model
            img_array = tf.keras.applications.resnet50.preprocess_input(img_array)

        else:
            return "Invalid Model"

        # ===============================
        # PREDICTION
        # ===============================
        preds = model.predict(img_array)[0]

        class_idx = int(np.argmax(preds))
        confidence = preds[class_idx]
        result = label_map[str(class_idx)]

        print("Preds:", preds)
        print("Predicted:", result)

        # ===============================
        # SAVE GRAPH
        # ===============================
        classes = list(label_map.values())
        plt.figure()
        plt.bar(classes, preds)
        plt.xticks(rotation=30)
        plt.title("Prediction Confidence")

        graph_path = os.path.join("static/uploads", "graph.png")
        plt.savefig(graph_path)
        plt.close()

        # ===============================
        # GRAD-CAM
        # ===============================
        heatmap = make_gradcam_heatmap(img_array_raw, model, model_choice)

        if heatmap is not None:
            gradcam_image = overlay_heatmap(heatmap, original_image)
            gradcam_path = os.path.join("static/uploads", "gradcam.jpg")
            cv2.imwrite(gradcam_path, gradcam_image)
        else:
            gradcam_path = filepath

        return render_template(
            "result.html",
            prediction=result,
            confidence=round(confidence * 100, 2),
            image=filepath,
            gradcam=gradcam_path,
            graph=graph_path
        )

    except Exception as e:
        return f"Error: {str(e)}"

# ===============================
if __name__ == "__main__":
    app.run(debug=True)