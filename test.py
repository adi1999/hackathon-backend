def jac_loss(class_weights=[1,20]):
    def loss(y_true, y_pred):
        smooth = 100
        inter = tf.keras.backend.sum(tf.math.abs(y_true * y_pred), axis=-1)
        sum_value = tf.keras.backend.sum(tf.math.abs(y_true) + tf.math.abs(y_pred), axis=-1)
        jac_coeff = (inter + smooth)/(sum_value - inter + smooth)
        return (1 - jac_coeff) * smooth
    if class_weights:
        return weighted_loss(loss, class_weights)
    else:
        return loss