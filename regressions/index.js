require("@tensorflow/tfjs-node");
const tf = require("@tensorflow/tfjs");
const loadDSV = require("./load-csv");
const LinearRegression = require("./linear-regression");

let { features, labels, testFeatures, testLabels } = loadDSV("./cars.csv", {
  shuffle: true,
  splitTest: 50,
  dataColumns: ["horsepower"],
  labelColumns: ["mpg"],
});

const lr = new LinearRegression(features, labels, { learningRate: 0.0001, iterations: 100 });

lr.train();

const r2 = lr.test(testFeatures, testLabels);

console.log("R2 = ", r2);

console.log("Weights:\n\tm =", lr.weights.get(1, 0), "\n\tb =", lr.weights.get(0, 0));
