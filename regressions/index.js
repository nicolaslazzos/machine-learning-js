require("@tensorflow/tfjs-node");
const tf = require("@tensorflow/tfjs");
const loadDSV = require("./load-csv");
const LinearRegression = require('./linear-regression');

let { features, labels, testFeatures, testLabels } = loadDSV("./cars.csv", {
  shuffle: true,
  splitTest: 50,
  dataColumns: ["horsepower"],
  labelColumns: ["mpg"],
});

const lr = new LinearRegression(features, labels, { learningRate: 0.001, iterations: 100 });

lr.train();

console.log(lr.m, lr.b);