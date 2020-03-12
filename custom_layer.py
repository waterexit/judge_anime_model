from keras.layers import Layer
from keras import backend as K
from keras.engine import base_layer_utils

class CenterLossLayer(Layer):

    def __init__(self,  **kwargs):
        super().__init__(**kwargs)
        self.alpha = 0.5

    def build(self, input_shape):
        self.centers = self.add_weight(name='centers',
                                       shape=(2, 2),
                                       initializer='uniform',
                                       trainable=False)
        # self.counter = self.add_weight(name='counter',
        #                                shape=(1,),
        #                                initializer='zeros',
        #                                trainable=False)  # just for debugging
        super().build(input_shape)

    def call(self, x, mask=None):

        # x[0] is Nx2, x[1] is Nx10 onehot, self.centers is 10x2
        delta_centers = K.dot(K.transpose(x[1]), (K.dot(x[1], self.centers) - x[0]))  # 10x2
        center_counts = K.sum(K.transpose(x[1]), axis=1, keepdims=True) + 1  # 10x1
        delta_centers /= center_counts
        new_centers = self.centers - self.alpha * delta_centers
        if base_layer_utils.call_context().training:
            self.add_update(lambda:self.centers.assign(new_centers))
        # self.add_update((self.counter, self.counter + 1), x)

        self.result = x[0] - K.dot(x[1], self.centers)
        self.result = K.sum(self.result ** 2, axis=1, keepdims=True) #/ K.dot(x[1], center_counts)
        return self.result # Nx1
    
    # def get_config(self):
    #     config = {
    #         "alpha" : self.alpha
    #     }
    #     base_config = super().get_config()
    #     return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return K.int_shape(self.result)


### custom loss

def zero_loss(y_true, y_pred):
    return 0.5 * K.sum(y_pred, axis=0)
