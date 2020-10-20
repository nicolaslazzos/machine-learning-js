require("@tensorflow/tfjs-node");
const tf = require("@tensorflow/tfjs");
const loadCSV = require("./load-csv");

function knn(features, labels, predictionPoint, k) {
  const { mean, variance } = tf.moments(features, 0);

  const scaledPrediction = predictionPoint.sub(mean).div(variance.pow(0.5));

  return (
    features
      .sub(mean)
      .div(variance.pow(0.5))
      .sub(scaledPrediction)
      .pow(2)
      .sum(1)
      .pow(0.5)
      .expandDims(1)
      .concat(labels, 1)
      .unstack()
      .sort((a, b) => (a.get(0) > b.get(0) ? 1 : -1))
      .slice(0, k)
      .reduce((acc, pair) => acc + pair.get(1), 0) / k
  );
}

let { features, labels, testFeatures, testLabels } = loadCSV("kc_house_data.csv", {
  shuffle: true,
  splitTest: 10,
  dataColumns: ["lat", "long", "sqft_lot", "sqft_living"],
  labelColumns: ["price"],
});

features = tf.tensor(features);
labels = tf.tensor(labels);

testFeatures.forEach((testPoint, i) => {
  const predictedPrice = knn(features, labels, tf.tensor(testPoint), 10);

  const realPrice = testLabels[i][0];

  const error = (realPrice - predictedPrice) / realPrice;

  console.log("REAL:", predictedPrice, "PREDICTED:", realPrice, "ERROR:", error);
});

// node --inspect-brk index.js
// in chrome navigate to about:inspect
