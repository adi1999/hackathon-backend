def weighted_loss(original_loss_func, weights_list):
    def loss_func(true, pred):
        axis = -1
        class_selectors = tf.keras.backend.cast(tf.keras.backend.argmax(true, axis=axis), tf.int32)
        class_selectors = [tf.keras.backend.equal(i, class_selectors) for i in range(len(weights_list))]
        class_selectors = [tf.keras.backend.cast(x, tf.keras.backend.floatx()) for x in class_selectors]
        weights = [sel * w for sel, w in zip(class_selectors, weights_list)]
        weight_multiplier = weights[0]
        for i in range(1, len(weights)):
            weight_multiplier = weight_multiplier + weights[i]
        loss = original_loss_func(true, pred)
        loss = loss * weight_multiplier
        return loss
    return loss_func