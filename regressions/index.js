require("@tensorflow/tfjs-node");
const tf = require("@tensorflow/tfjs");
const loadDSV = require("./load-csv");
const LinearRegression = require("./linear-regression");
const plot = require("node-remote-plot");

let { features, labels, testFeatures, testLabels } = loadDSV("./cars.csv", {
  shuffle: true,
  splitTest: 50,
  dataColumns: ["horsepower", "weight", "displacement"],
  labelColumns: ["mpg"],
});

const lr = new LinearRegression(features, labels, { learningRate: 0.1, iterations: 3, batchSize: 10 });

lr.train();

const r2 = lr.test(testFeatures, testLabels);

console.log("R2 = ", r2);

// mse history
plot({ x: lr.mseHistory.reverse(), xLabel: "Iteration #", yLabel: "MSE" });

lr.predict([[120, 2, 380]]).print();
