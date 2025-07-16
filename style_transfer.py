import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import vgg19
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import PIL.Image

def load_and_process_image(image_path):
    img = load_img(image_path, target_size=(512, 512), color_mode='rgb')  # Ensure 3 channels
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return vgg19.preprocess_input(img)

def deprocess_img(processed_img):
    x = processed_img.copy().squeeze()  # remove batch dim → (H, W, 3)
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]  # BGR to RGB
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def get_model():
    vgg = vgg19.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    style_layers = [
        'block1_conv1',
        'block2_conv1',
        'block3_conv1',
        'block4_conv1',
        'block5_conv1'
    ]
    content_layers = ['block5_conv2']
    output_layers = style_layers + content_layers
    model = Model([vgg.input], [vgg.get_layer(name).output for name in output_layers])
    return model, style_layers, content_layers

def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result / num_locations

def get_content_loss(base_content, target):
    return tf.reduce_mean(tf.square(base_content - target))

def get_style_loss(base_style, gram_target):
    gram_style = gram_matrix(base_style)

    # Debug print (optional)
    print("gram_style shape:", gram_style.shape)
    print("gram_target shape:", gram_target.shape)

    if gram_style.shape != gram_target.shape:
        raise ValueError(f"Shape mismatch in style loss: gram_style {gram_style.shape}, gram_target {gram_target.shape}")

    return tf.reduce_mean(tf.square(gram_style - gram_target))

def perform_style_transfer(content_path, style_path, num_iterations=100, content_weight=1e4, style_weight=1e-1):
    model, style_layers, content_layers = get_model()
    content_img = load_and_process_image(content_path)
    style_img = load_and_process_image(style_path)
    generated_img = tf.Variable(content_img, dtype=tf.float32)

    def get_feature_representations():
        style_outputs = model(style_img)
        content_outputs = model(content_img)
        style_features = [gram_matrix(layer) for layer in style_outputs[:len(style_layers)]]
        content_features = [layer for layer in content_outputs[len(style_layers):]]
        return style_features, content_features

    def compute_loss(generated_img, style_features, content_features):
        model_outputs = model(generated_img)
        gen_style = model_outputs[:len(style_layers)]
        gen_content = model_outputs[len(style_layers):]

        style_loss = tf.add_n([get_style_loss(gs, sf) for gs, sf in zip(gen_style, style_features)])
        content_loss = tf.add_n([get_content_loss(gc, cf) for gc, cf in zip(gen_content, content_features)])

        total_loss = content_weight * content_loss + style_weight * style_loss
        return total_loss

    style_features, content_features = get_feature_representations()
    optimizer = tf.optimizers.Adam(learning_rate=5.0)

    for i in range(num_iterations):
        with tf.GradientTape() as tape:
            loss = compute_loss(generated_img, style_features, content_features)
        grad = tape.gradient(loss, generated_img)
        optimizer.apply_gradients([(grad, generated_img)])
        generated_img.assign(tf.clip_by_value(generated_img, -128, 128))

    final_img = deprocess_img(generated_img.numpy())
    return final_img
