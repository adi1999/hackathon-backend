def conv_bn_relu_x2(nb_filter, row, col, kernel_initializer='glorot_uniform', kernel_regularizers=None,
                    bias_regularizers=None, dropout=False):
    def f(input):
        conv_a = Conv2D(nb_filter, (row, col),
                        kernel_initializer=kernel_initializer, padding='same', use_bias=False,
                        kernel_regularizer=kernel_regularizers,
                        bias_regularizer=bias_regularizers, trainable=True)(input)
        norm_a = BatchNormalization(trainable=True)(conv_a)
        act_a = Activation(activation='relu')(norm_a)
        if dropout:
            act_a = Dropout(0.5)(act_a)
        conv_b = Conv2D(nb_filter, (row, col),
                        kernel_initializer=kernel_initializer, padding='same', use_bias=False,
                        kernel_regularizer=kernel_regularizers,
                        bias_regularizer=bias_regularizers, trainable=True)(act_a)
        norm_b = BatchNormalization(trainable=True)(conv_b)
        act_b = Activation(activation='relu')(norm_b)
        if dropout:
            act_b = Dropout(0.5)(act_b)
        return act_b
    return f