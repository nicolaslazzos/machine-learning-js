const tf = require("@tensorflow/tfjs");
const _ = require("lodash");

class LinearRegression {
  constructor(features, labels, options = {}) {
    // features
    this.features = this.processFeatures(features);
    // labels
    this.labels = tf.tensor(labels);
    // weights
    this.weights = tf.zeros([this.features.shape[1], 1]);
    // options
    this.options = Object.assign({ learningRate: 0.1, iterations: 1000, batchSize: this.features.shape[0] }, options);
    // mse records in each interation
    this.mseHistory = [];
  }

  train() {
    const batchQuantity = Math.floor(this.features.shape[0] / this.options.batchSize);

    for (let i = 0; i < this.options.iterations; i++) {
      for (let j = 0; j < batchQuantity; j++) {
        const startIndex = [j * this.options.batchSize, 0];
        const endIndex = [this.options.batchSize, -1];

        const featuresSlice = this.features.slice(startIndex, endIndex);
        const labelsSlice = this.labels.slice(startIndex, endIndex);

        this.gradientDescent(featuresSlice, labelsSlice);
      }
      this.mse();
      this.updateLearningRate();
    }
  }

  processFeatures(features) {
    features = tf.tensor(features);
    features = this.standardize(features);
    return tf.ones([features.shape[0], 1]).concat(features, 1);
  }

  standardize(features) {
    if (!this.mean && !this.variance) {
      const { mean, variance } = tf.moments(features, 0);

      this.mean = mean;
      this.variance = variance;
    }

    return features.sub(this.mean).div(this.variance.pow(0.5));
  }

  gradientDescent(features, labels) {
    // m * x + b
    const currentGuesses = features.matMul(this.weights);

    // guess - real
    const differences = currentGuesses.sub(labels);

    // [bSlope, mSlope]
    const slopes = features.transpose().matMul(differences).div(features.shape[0]);

    // update weights
    this.weights = this.weights.sub(slopes.mul(this.options.learningRate));

    // const mpgGuesses = this.features.map((row) => this.m * row[0] + this.b);
    // const bSlope = (_.sum(mpgGuesses.map((guess, i) => guess - this.labels[i][0])) * 2) / this.features.length;
    // const mSlope =
    //   (_.sum(mpgGuesses.map((guess, i) => -1 * this.features[i][0] * (this.labels[i][0] - guess))) * 2) /
    //   this.features.length;

    // this.m = this.m - mSlope * this.options.learningRate;
    // this.b = this.b - bSlope * this.options.learningRate;
  }

  mse() {
    const mse = this.features.matMul(this.weights).sub(this.labels).pow(2).sum().div(this.features.shape[0]).get();
    this.mseHistory.unshift(mse);
  }

  updateLearningRate() {
    if (this.mseHistory.length < 2) return;

    const last = this.mseHistory[0];
    const prev = this.mseHistory[1];

    if (last > prev) {
      this.options.learningRate /= 2;
    } else {
      this.options.learningRate *= 1.05;
    }
  }

  test(features, labels) {
    // test features
    features = this.processFeatures(features);
    // test labels
    labels = tf.tensor(labels);

    // predictions
    const predictions = features.matMul(this.weights);

    // sum of squares residuals
    const res = labels.sub(predictions).pow(2).sum().get();
    // sum of squares total
    const tot = labels.sub(labels.mean()).pow(2).sum().get();

    // coefficient of determination (r2)
    return 1 - res / tot;
  }

  predict(observations) {
    return this.processFeatures(observations).matMul(this.weights);
  }
}

module.exports = LinearRegression;
