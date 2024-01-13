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


def weighted_cc(class_weights):
    def loss(y_true, y_pred):
        cc_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        return cc_los
    return weighted_loss(loss, class_weights)


def count_plus_mse_loss_channelwise2(loss_weights=(1, 1), dm_weight=100):
    def loss(y_true, y_pred):
        overall_loss = []
        for channel in range(y_pred.shape[-1]):
            pred_count = tf.round(tf.reduce_sum(y_pred[:, :, :, channel]) / dm_weight)
            true_count = tf.round(tf.reduce_sum(y_true[:, :, :, channel]) / dm_weight)
            count_loss = tf.square(tf.subtract(pred_count, true_count))
            count_loss = tf.multiply(count_loss, loss_weights[0])  #(1, no_clas)
            mse_loss = tf.square(tf.subtract(y_true[:, :, :, channel], y_pred[:, :, :, channel]))
            mse_loss = tf.multiply(mse_loss, loss_weights[1])  # (320, 320)
            overall_loss.append(tf.multiply(mse_loss, count_loss))
        # overall_loss = tf.reduce_mean(overall_loss, axis=0)
        overall_loss = tf.stack(overall_loss)
        overall_loss = tf.transpose(overall_loss, perm=[1, 0, 2, 3])
        overall_loss = tf.transpose(overall_loss, perm=[0, 3, 2, 1])
        overall_loss = tf.transpose(overall_loss, perm=[0, 2, 1, 3])  # (320 ,320, no_classes)
        return overall_loss
    return loss