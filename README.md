# spam_email_classifier
Simple spam email classifier was implemented by logistic regression, neural network and svm.

## install third party library

## how to use
`    def train(self, context):
        features, labels = context.get_samples()
        self._features_count = features.shape[1]
        self._theta_size = self._features_count * self._hidden_layer_units + \
                            (self._hidden_layer_units + 1) * 1
        self._initialize_all()
        # checking whether gradient is correct,
        # compare it with numerical gradient
        # logging.info('compute back propagation once to check gradient')
        # self._back_propagation(features, labels)
        # if not self._checking_gradient(features, labels):
        #     logging.info('gradient is not equal to numerical value approximatly')
        #     return
        # use gradient descent to compute theta
        for i in range(self._iterations):
            self._back_propagation(features, labels)
            for j in range(self._theta_size):
                self._theta[j] = self._theta[j] + self._alpha*self._gradient[j]
            cost = self._cost_function(features, labels, self._theta)
            logging.info('iteration: %d, cost: %f' % (i+1, cost))`
