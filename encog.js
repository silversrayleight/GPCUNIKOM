/*
 * http://code.google.com/p/encog-java/
 */

/**
 * The main Encog namespace.  This is the only global property created by Encog.
 * @type {*}
 */
var ENCOG = ENCOG || {
    VERSION : '0.1',
    PLATFORM : 'javascript',
    precision : 1e-10,
    NEWLINE : '\n',
    ENCOG_TYPE_ACTIVATION : 'ActivationFunction',
    ENCOG_TYPE_RBF : 'RBF'
};
ENCOG.namespace = function (namespaceString) {
    'use strict';

    var parts = namespaceString.split('.'),
        parent = window,
        currentPart = '',
        i,
        length;

    for (i = 0, length = parts.length; i < length; i += 1) {
        currentPart = parts[i];
        parent[currentPart] = parent[currentPart] || {};
        parent = parent[currentPart];
    }

    return parent;
};

ENCOG.namespace('ENCOG.ActivationSigmoid');
ENCOG.namespace('ENCOG.ActivationTANH');
ENCOG.namespace('ENCOG.ActivationLinear');
ENCOG.namespace('ENCOG.ActivationElliott');
ENCOG.namespace('ENCOG.ActivationElliottSymmetric');
ENCOG.namespace('ENCOG.RadialGaussian');
ENCOG.namespace('ENCOG.RadialMexicanHat');
ENCOG.namespace('ENCOG.Util');
ENCOG.namespace('ENCOG.MathUtil');
ENCOG.namespace('ENCOG.ArrayUtil');
ENCOG.namespace('ENCOG.BasicLayer');
ENCOG.namespace('ENCOG.BasicNetwork');
ENCOG.namespace('ENCOG.PropagationTrainer');
ENCOG.namespace('ENCOG.LinearErrorFunction');
ENCOG.namespace('ENCOG.LinearErrorFunction');
ENCOG.namespace('ENCOG.Swarm');
ENCOG.namespace('ENCOG.Anneal');
ENCOG.namespace('ENCOG.Genetic');
ENCOG.namespace('ENCOG.SOM');
ENCOG.namespace('ENCOG.TrainSOM');
ENCOG.namespace('ENCOG.ReadCSV');
ENCOG.namespace('ENCOG.EGFILE');

ENCOG.MathUtil = function () {
    'use strict';
};
ENCOG.MathUtil.tanh = function (x) {
    'use strict';
    var pos, neg;

    pos = Math.exp(x);
    neg = Math.exp(-x);

    return (pos - neg) / (pos + neg);

};
ENCOG.MathUtil.sign = function (x) {
    'use strict';
    if (Math.abs(x) < ENCOG.precision) {
        return 0;
    } else if (x > 0) {
        return 1;
    } else {
        return -1;
    }
};

ENCOG.MathUtil.euclideanDistance = function (a1, a2, startIndex, len) {
    'use strict';

    var result = 0, i, diff;
    for (i = startIndex; i < (startIndex + len); i += 1) {
        diff = a1[i] - a2[i];
        result += diff * diff;
    }
    return Math.sqrt(result);
};

ENCOG.MathUtil.kNearest = function (a1, lst, k, maxDist, startIndex, len) {
    'use strict';
    var result = [], tempDist = [], idx = 0, worstIdx = -1, dist, agent;

    while (idx < lst.length) {
        agent = lst[idx];
        if (a1 !== agent) {
            dist = ENCOG.MathUtil.euclideanDistance(a1, agent, startIndex, len);

            if (dist < maxDist) {
                if (result.length < k) {
                    result.push(agent);
                    tempDist.push(dist);
                    worstIdx = ENCOG.ArrayUtil.arrayMaxIndex(tempDist);
                } else {
                    if (dist < tempDist[worstIdx]) {
                        tempDist[worstIdx] = dist;
                        result[worstIdx] = agent;
                        worstIdx = ENCOG.ArrayUtil.arrayMaxIndex(tempDist);
                    }
                }
            }
        }

        idx += 1;
    }

    return result;
};

ENCOG.MathUtil.randomFloat = function (low, high) {
    'use strict';
    return (Math.random * (high - low)) + low;
};

ENCOG.ArrayUtil = function () {
    'use strict';
};
ENCOG.ArrayUtil.fillArray = function (arr, start, stop, v) {
    'use strict';
    var i;

    for (i = start; i < stop; i += 1) {
        arr[i] = v;
    }
};

ENCOG.ArrayUtil.newFloatArray = function (sz) {
    'use strict';
    var result;
    result = [];
    while (sz > 0) {
        result.push(0.0);
        sz-=1;
    }
    return result;
};
ENCOG.ArrayUtil.newIntArray = function (sz) {
    'use strict';
    var result;
    result = [];
    while ((sz -= 1) > 0) {
        result.push(0);
    }
    return result;
};

ENCOG.ArrayUtil.fillArray2D = function (arr, v) {
    'use strict';
    var i, j, row;

    for (i = 0; i < arr.length; i += 1) {
        row = arr[i];
        for (j = 0; j < row.length; j += 1) {
            row[j] = v;
        }
    }
};

ENCOG.ArrayUtil.randomizeArray = function (arr, start, stop, low, high) {
    'use strict';
    var i;

    for (i = start; i < stop; i += 1) {
        arr[i] = ENCOG.MathUtil.randomFloat(low, high);
    }
};

ENCOG.ArrayUtil.randomizeArray2D = function (arr, low, high) {
    'use strict';
    var i, j, row;

    for (i = 0; i < arr.length; i += 1) {
        row = arr[i];
        for (j = 0; j < row.length; j += 1) {
            row[j] = ENCOG.MathUtil.randomFloat(low, high);
        }
    }
};

ENCOG.ArrayUtil.allocate1D = function (x) {
    'use strict';
    var i, result;

    result = [];
    for (i = 0; i < x; i += 1) {
        result[i] = 0;
    }

    return result;
};

ENCOG.ArrayUtil.allocateBoolean2D = function (rows, cols) {
    'use strict';
    var result, row, col, temp;

    result = [
        []
    ];

    for (row = 0; row < rows; row += 1) {
        temp = [];
        for (col = 0; col < cols; col += 1) {
            temp[col] = false;
        }
        result[row] = temp;
    }

    return result;
};

ENCOG.ArrayUtil.arrayCopy = function (source, sourceStart, target, targetStart, count) {
    'use strict';
    var i;

    for (i = 0; i < count; i += 1) {
        target[i + targetStart] = source[i + sourceStart];
    }
};

ENCOG.ArrayUtil.generateBenchmarkData = function (rowCount, colCount) {
    'use strict';
    var result, item, row, col;

    result = [
        []
    ];

    for (row = 0; row < rowCount; row += 1) {
        item = [];
        for (col = 0; col < colCount; col += 1) {
            item[col] = (Math.random() * 2) - 1;
        }
        result[row] = item;
    }

    return result;
};

ENCOG.ArrayUtil.arrayMean = function (a1, idx) {
    'use strict';
    var result, i;

    result = 0;
    for (i = 0; i < a1.length; i += 1) {
        result += a1[i][idx];
    }
    result /= a1.length;
    return result;
};
ENCOG.ArrayUtil.arrayMinIndex = function (a1) {
    'use strict';
    var result, resultIndex, i;

    result = Number.MAX_VALUE;
    resultIndex = -1;

    for (i = 0; i < a1.length; i += 1) {
        if (a1[i] < result) {
            result = a1[i];
            resultIndex = i;
        }
    }
    return resultIndex;
};
ENCOG.ArrayUtil.arrayMaxIndex = function (a1) {
    'use strict';
    var result, resultIndex, i;

    result = Number.MIN_VALUE;
    resultIndex = -1;

    for (i = 0; i < a1.length; i += 1) {
        if (a1[i] > result) {
            result = a1[i];
            resultIndex = i;
        }
    }
    return resultIndex;
};

ENCOG.Util.stripQuotes = function (s) {
    'use strict';
    var l = s.length;
    if (s[0] === '\"' || s[0] === '\'') {
        s = s.substr(1);
        l -= 1;
    }

    if (s[l - 1] === '\"' || s[l - 1] === '\'') {
        s = s.substr(0, l - 1);
    }

    return s;
};

ENCOG.ActivationSigmoid = function () {
    'use strict';
};
ENCOG.ActivationSigmoid.prototype = {
    NAME : "ActivationSigmoid",

      encogType : ENCOG.ENCOG_TYPE_ACTIVATION,

      activationFunction : function (x, start, size) {
        'use strict';
        var i;

        for (i = start; i < start + size; i += 1) {
            x[i] = 1.0 / (1.0 + Math.exp(-1 * x[i]));
        }
    },
    derivativeFunction : function (b, a) {
        'use strict';
        return a * (1.0 - a);
    }
};

ENCOG.ActivationSigmoid.create = function () {
    'use strict';
    return new ENCOG.ActivationSigmoid();
};

ENCOG.ActivationTANH = function () {
    'use strict';
};

ENCOG.ActivationTANH.prototype = {
   
    NAME : "ActivationTANH",
    encogType : ENCOG.ENCOG_TYPE_ACTIVATION,

    activationFunction : function (x, start, size) {
        'use strict';
        var i;

        for (i = start; i < start + size; i += 1) {
            x[i] = ENCOG.MathUtil.tanh(x[i]);
        }
    },
    derivativeFunction : function (b, a) {
        'use strict';
        return (1.0 - a * a);
    }
};

ENCOG.ActivationTANH.create = function () {
    'use strict';
    return new ENCOG.ActivationTANH();
};

ENCOG.ActivationLinear = function () {
    'use strict';
};
ENCOG.ActivationLinear.prototype = {

    NAME : "ActivationLinear",

    
    encogType : ENCOG.ENCOG_TYPE_ACTIVATION,

    activationFunction : function () {
        'use strict';
    },

   
    derivativeFunction : function () {
        'use strict';
        return 1.0;
    }
};

ENCOG.ActivationLinear.create = function () {
    'use strict';
    return new ENCOG.ActivationLinear();
};

ENCOG.ActivationElliott = function () {
    'use strict';
};

ENCOG.ActivationElliott.prototype = {

    NAME : "ActivationElliott",

    
    encogType : ENCOG.ENCOG_TYPE_ACTIVATION,
    slope : 1,

    
    activationFunction : function (x, start, size) {
        'use strict';
        var i;

        for (i = start; i < start + size; i += 1) {
            x[i] = ((x[i] * this.slope) / 2) / (1 + Math.abs(x[i] * this.slope)) + 0.5;
        }
    },

   
    derivativeFunction : function (b, a) {
        'use strict';
        return this.slope / (2.0 * (1.0 + Math.abs(b * this.slope)) * (1 + Math.abs(b * this.slope)));
    }
};

ENCOG.ActivationElliott.create = function (s) {
    'use strict';
    var result = new ENCOG.ActivationElliott();
    result.slope = s || 1;
    return result;
};

ENCOG.ActivationElliottSymmetric = function () {
    'use strict';
};

ENCOG.ActivationElliottSymmetric.prototype = {

    
    NAME : "ActivationElliottSymmetric",

   
    encogType : ENCOG.ENCOG_TYPE_ACTIVATION,
    slope : 1,

    
    activationFunction : function (x, start, size) {
        'use strict';
        var i;
        for (i = start; i < start + size; i += 1) {
            x[i] = (x[i] * this.slope) / (1 + Math.abs(x[i] * this.slope));
        }
    },

    
    derivativeFunction : function (b, a) {
        'use strict';
        var d = (1.0 + Math.abs(b * this.slope));
        return this.slope / (d * d);
    }
};

ENCOG.ActivationElliottSymmetric.create = function (s) {
    'use strict';
    var result = new ENCOG.ActivationElliottSymmetric();
    result.slope = s || 1;
    return result;
};

ENCOG.RadialGaussian = function () {
    'use strict';
};

ENCOG.RadialGaussian.prototype = {

    NAME : "RadialGaussian",

    
    encogType : ENCOG.ENCOG_TYPE_RBF,
    center : [],
    width : 1,
    peak : 1,

   
    calculate : function (x) {
        'use strict';

        var value = 0, i;

        for (i = 0; i < this.center.length; i += 1) {
            value += Math.pow(x[i] - this.center[i], 2) / (2.0 * this.width * this.width);
        }
        return this.peak * Math.exp(-value);
    }
};

ENCOG.RadialGaussian.create = function (thePeak, theCenters, theWidth) {
    'use strict';
    var result = new ENCOG.RadialGaussian();
    result.peak = thePeak || 1;
    result.centers = theCenters;
    result.width = theWidth || 1;
    return result;
};

ENCOG.RadialMexicanHat = function () {
    'use strict';
};

ENCOG.RadialMexicanHat.prototype = {

    NAME : "RadialMexicanHat",

    
    encogType : ENCOG.ENCOG_TYPE_RBF,
    center : [],
    width : [],
    peak : 1,

    calculate : function (x) {
        'use strict';

        var norm = 0, i;
        for (i = 0; i < this.center.length; i += 1) {
            norm += Math.pow(x[i] - this.center[i], 2);
        }

        return this.peak * (1 - norm) * Math.exp(-norm / 2);
    }
};


ENCOG.RadialMexicanHat.create = function (thePeak, theCenters, theWidth) {
    'use strict';
    var result = new ENCOG.RadialMexicanHat();
    result.peak = thePeak || 1;
    result.centers = theCenters;
    result.width = theWidth || 1;
    return result;
};

ENCOG.LinearErrorFunction = function () {
    'use strict';
};

ENCOG.LinearErrorFunction.prototype = {
    
    calculateError : function (ideal, actual, error) {
        'use strict';
        var i;
        for (i = 0; i < actual.length; i += 1) {
            error[i] = ideal[i] - actual[i];
        }

    }
};

ENCOG.LinearErrorFunction.create = function () {
    'use strict';
    return new ENCOG.LinearErrorFunction();
};

ENCOG.BasicLayer = function () {
    'use strict';
};

ENCOG.BasicLayer.prototype = {

    NAME : 'BasicLayer',

   
    activation : null,

   
    count : null,

    biasActivation : null,

    
    contextFedBy : null,

    calcTotalCount : function () {
        'use strict';
        if (this.contextFedBy === null) {
            return this.count + (this.hasBias() ? 1 : 0);
        } else {
            return this.count + (this.hasBias() ? 1 : 0)
                + this.contextFedBy.count;
        }
    },

   
    hasBias : function () {
        'use strict';
        return Math.abs(this.biasActivation) > ENCOG.precision;
    },

    
    calcContextCount : function () {
        'use strict';
        if (this.contextFedBy === null) {
            return 0;
        } else {
            return this.contextFedBy.count;
        }
    }
};

ENCOG.BasicLayer.create = function (activation, count, biasActivation) {
    'use strict';
    var result;

    if (activation.encogType !== ENCOG.ENCOG_TYPE_ACTIVATION) {
        throw new Error("Invalid activation function.");
    }
    result = new ENCOG.BasicLayer();
    result.activation = activation;
    result.count = count;
    result.biasActivation = biasActivation;
    result.contextFedBy = null;
    return result;
};

ENCOG.BasicNetwork = function () {
    'use strict';
};
ENCOG.BasicNetwork.prototype = {
    
    NAME : 'BasicNetwork',

    
    inputCount : null,

   
    outputCount : null,

   
    layerCounts : null,

   
    layerContextCount : null,

   
    weightIndex : null,

   
    layerIndex : null,

   
    activationFunctions : null,

   
    layerFeedCounts : null,

   
    contextTargetOffset : null,

   
    contextTargetSize : null,

    
    biasActivation : null,

    
    beginTraining : null,

    
    endTraining : null,

   
    weights : null,

    layerOutput : null,

    
    layerSums : null,

    connectionLimit : ENCOG.precision,

    clearContext : function () {
        'use strict';
        var index, i, hasBias;

        index = 0;
        for (i = 0; i < this.layerIndex.length; i += 1) {

            hasBias = (this.layerContextCount[i] + this.layerFeedCounts[i]) !== this.layerCounts[i];

            // fill in regular neurons
            ENCOG.ArrayUtil.fillArray(this.layerOutput, index, index + this.layerFeedCounts[i], 0);
            index += this.layerFeedCounts[i];

            // fill in the bias
            if (hasBias) {
                this.layerOutput[index] = this.biasActivation[i];
                index += 1;
            }

            // fill in context
            ENCOG.ArrayUtil.fillArray(this.layerOutput, index, index + this.layerContextCount[i], 0);
            index += this.layerContextCount[i];
        }
    },

    randomize : function () {
        'use strict';
        var i;
        for (i = 0; i < this.weights.length; i += 1) {
            this.weights[i] = (Math.random() * 2.0) - 1.0;
        }
    },

    computeLayer : function (currentLayer) {
        'use strict';
        var inputIndex, outputIndex, inputSize, outputSize, index, limitX, limitY,
            x, sum, offset, y;

        inputIndex = this.layerIndex[currentLayer];
        outputIndex = this.layerIndex[currentLayer - 1];
        inputSize = this.layerCounts[currentLayer];
        outputSize = this.layerFeedCounts[currentLayer - 1];

        index = this.weightIndex[currentLayer - 1];

        limitX = outputIndex + outputSize;
        limitY = inputIndex + inputSize;

        // weight values
        for (x = outputIndex; x < limitX; x += 1) {
            sum = 0;
            for (y = inputIndex; y < limitY; y += 1) {
                sum += this.weights[index] * this.layerOutput[y];
                index += 1;
            }
            this.layerSums[x] = sum;
            this.layerOutput[x] = sum;
        }

        this.activationFunctions[currentLayer - 1].activationFunction(
            this.layerOutput,
            outputIndex,
            outputSize
        );

        // update context values
        offset = this.contextTargetOffset[currentLayer];

        ENCOG.ArrayUtil.arrayCopy(this.layerOutput, outputIndex,
            this.layerOutput, offset, this.contextTargetSize[currentLayer]);
    },

    compute : function (input, output) {
        'use strict';
        var sourceIndex, i, offset;

        sourceIndex = this.layerOutput.length
            - this.layerCounts[this.layerCounts.length - 1];

        ENCOG.ArrayUtil.arrayCopy(input, 0, this.layerOutput, sourceIndex,
            this.inputCount);

        for (i = this.layerIndex.length - 1; i > 0; i -= 1) {
            this.computeLayer(i);
        }

        // update context values
        offset = this.contextTargetOffset[0];

        ENCOG.ArrayUtil.arrayCopy(this.layerOutput, 0, this.layerOutput,
            offset, this.contextTargetSize[0]);

        ENCOG.ArrayUtil.arrayCopy(this.layerOutput, 0, output, 0, this.outputCount);
    },
    evaluate : function (inputData, idealData) {
        'use strict';
        var i, j, input, ideal, output, diff, globalError, setSize;

        output = [];
        globalError = 0;
        setSize = 0;

        for (i = 0; i < inputData.length; i += 1) {
            input = inputData[i];
            ideal = idealData[i];
            this.compute(input, output);
            for (j = 0; j < ideal.length; j += 1) {
                diff = ideal[j] - output[j];
                globalError += diff * diff;
                setSize += 1;
            }
        }

        return globalError / setSize;
    }


};


ENCOG.BasicNetwork.create = function (layers) {
    'use strict';
    var layerCount, result, index, neuronCount, weightCount, i, j, layer, nextLayer, neuronIndex;

    result = new ENCOG.BasicNetwork();
    layerCount = layers.length;

    result.inputCount = layers[0].count;
    result.outputCount = layers[layerCount - 1].count;

    result.layerCounts = ENCOG.ArrayUtil.allocate1D(layerCount);
    result.layerContextCount = ENCOG.ArrayUtil.allocate1D(layerCount);
    result.weightIndex = ENCOG.ArrayUtil.allocate1D(layerCount);
    result.layerIndex = ENCOG.ArrayUtil.allocate1D(layerCount);
    result.activationFunctions = ENCOG.ArrayUtil.allocate1D(layerCount);
    result.layerFeedCounts = ENCOG.ArrayUtil.allocate1D(layerCount);
    result.contextTargetOffset = ENCOG.ArrayUtil.allocate1D(layerCount);
    result.contextTargetSize = ENCOG.ArrayUtil.allocate1D(layerCount);
    result.biasActivation = ENCOG.ArrayUtil.allocate1D(layerCount);

    index = 0;
    neuronCount = 0;
    weightCount = 0;

    for (i = layers.length - 1; i >= 0; i -= 1) {
        layer = layers[i];
        nextLayer = null;

        if (i > 0) {
            nextLayer = layers[i - 1];
        }

        result.biasActivation[index] = layer.biasActivation;
        result.layerCounts[index] = layer.calcTotalCount();
        result.layerFeedCounts[index] = layer.count;
        result.layerContextCount[index] = layer.calcContextCount();
        result.activationFunctions[index] = layer.activation;

        neuronCount += layer.calcTotalCount();

        if (nextLayer !== null) {
            weightCount += layer.count * nextLayer.calcTotalCount();
        }

        if (index === 0) {
            result.weightIndex[index] = 0;
            result.layerIndex[index] = 0;
        } else {
            result.weightIndex[index] = result.weightIndex[index - 1]
                + (result.layerCounts[index] * result.layerFeedCounts[index - 1]);
            result.layerIndex[index] = result.layerIndex[index - 1]
                + result.layerCounts[index - 1];
        }

        neuronIndex = 0;
        for (j = layers.length - 1; j >= 0; j -= 1) {
            if (layers[j].contextFedBy === layer) {
                result.hasContext = true;
                result.contextTargetSize[index] = layers[j].calcContextCount();
                result.contextTargetOffset[index] = neuronIndex
                    + (layers[j].calcTotalCount() - layers[j]
                    .calcContextCount());
            }
            neuronIndex += layers[j].calcTotalCount();
        }

        index += 1;
    }

    result.beginTraining = 0;
    result.endTraining = result.layerCounts.length - 1;

    result.weights = ENCOG.ArrayUtil.allocate1D(weightCount);
    result.layerOutput = ENCOG.ArrayUtil.allocate1D(neuronCount);
    result.layerSums = ENCOG.ArrayUtil.allocate1D(neuronCount);

    result.clearContext();
    return result;
};


ENCOG.PropagationTrainer = function () {
    'use strict';
};

ENCOG.PropagationTrainer.prototype = {
   
    NAME : 'PropagationTrainer',
  
    POSITIVE_ETA : 1.2,

   
    NEGATIVE_ETA : 0.5,

    
    DELTA_MIN : 1e-6,

    MAX_STEP : 50,

    
    network : null,

   
    trainingInput : null,

   
    trainingIdeal : null,

   
    type : null,

  
    learningRate : null,

   
    momentum : null,

   
    layerDelta : null,

    
    gradients : null,

    
    lastGradient : null,

    
    lastDelta : null,

   
    actual : null,

   
    flatSpot : null,

    errorFunction : ENCOG.LinearErrorFunction.create(),

   
    updateValues : null,

    processLevel : function (currentLevel) {
        'use strict';
        var toLayerIndex, fromLayerIndex, index, fromLayerSize, toLayerSize, activation,
            currentFlatSpot, yi, output, sum, xi, wi, y, x;

        fromLayerIndex = this.network.layerIndex[currentLevel + 1];
        toLayerIndex = this.network.layerIndex[currentLevel];
        fromLayerSize = this.network.layerCounts[currentLevel + 1];
        toLayerSize = this.network.layerFeedCounts[currentLevel];

        index = this.network.weightIndex[currentLevel];
        activation = this.network.activationFunctions[currentLevel + 1];
        currentFlatSpot = this.flatSpot[currentLevel + 1];

        // handle weights
        yi = fromLayerIndex;
        for (y = 0; y < fromLayerSize; y += 1) {
            output = this.network.layerOutput[yi];
            sum = 0;
            xi = toLayerIndex;
            wi = index + y;
            for (x = 0; x < toLayerSize; x += 1) {
                this.gradients[wi] += output * this.layerDelta[xi];
                sum += this.network.weights[wi] * this.layerDelta[xi];
                wi += fromLayerSize;
                xi += 1;
            }

            this.layerDelta[yi] = sum
                * (activation.derivativeFunction(this.network.layerSums[yi], this.network.layerOutput[yi]) + currentFlatSpot);
            yi += 1;
        }
    },

    learnBPROP : function () {
        'use strict';
        var i, delta;

        for (i = 0; i < this.network.weights.length; i += 1) {
            delta = (this.gradients[i] * this.learningRate) + (this.lastDelta[i] * this.momentum);
            this.lastDelta[i] = delta;
            this.network.weights[i] += delta;
        }
    },

    learnRPROP : function () {
        'use strict';
        var delta, change, weightChange, i;

        for (i = 0; i < this.network.weights.length; i += 1) {
           
            change = ENCOG.MathUtil.sign(this.gradients[i] * this.lastGradient[i]);
            weightChange = 0;

            if (change > 0) {
                delta = this.updateValues[i]
                    * this.POSITIVE_ETA;
                delta = Math.min(delta, this.MAX_STEP);
                weightChange = ENCOG.MathUtil.sign(this.gradients[i]) * delta;
                this.updateValues[i] = delta;
                this.lastGradient[i] = this.gradients[i];
            } else if (change < 0) {
              
                delta = this.updateValues[i]
                    * this.NEGATIVE_ETA;
                delta = Math.max(delta, this.DELTA_MIN);
                this.updateValues[i] = delta;
                weightChange = -this.lastDelta[i];
                
                this.lastGradient[i] = 0;
            } else if (change === 0) {
               
                delta = this.updateValues[i];
                weightChange = ENCOG.MathUtil.sign(this.gradients[i]) * delta;
                this.lastGradient[i] = this.gradients[i];
            }

            this.network.weights[i] += weightChange;
        }
    },


    process : function (input, ideal, s) {
        'use strict';
        var i, j, delta;

        this.network.compute(input, this.actual);

        for (j = 0; j < ideal.length; j += 1) {
            delta = this.actual[j] - ideal[j];
            this.globalError = this.globalError + (delta * delta);
            this.setSize += 1;
        }

        this.errorFunction.calculateError(ideal, this.actual, this.layerDelta);

        for (i = 0; i < this.actual.length; i += 1) {
            this.layerDelta[i] = ((this.network.activationFunctions[0]
                .derivativeFunction(this.network.layerSums[i], this.network.layerOutput[i]) + this.flatSpot[0]))
                * (this.layerDelta[i] * s);
        }

        for (i = this.network.beginTraining; i < this.network.endTraining; i += 1) {
            this.processLevel(i);
        }
    },

    iteration : function () {
        'use strict';
        var i;
        this.globalError = 0;
        this.setSize = 0;
        this.actual = [];

        ENCOG.ArrayUtil.fillArray(this.gradients, 0, this.gradients.length, 0);
        ENCOG.ArrayUtil.fillArray(this.lastDelta, 0, this.lastDelta.length, 0);

        for (i = 0; i < this.trainingInput.length; i += 1) {
            this.process(this.trainingInput[i], this.trainingIdeal[i], 1.0);
        }

        if (this.type === "BPROP") {
            this.learnBPROP();
        } else if (this.type === "RPROP") {
            this.learnRPROP();
        }

        this.error = this.globalError / this.setSize;
    }

};

ENCOG.PropagationTrainer.create = function (network, input, ideal, type, learningRate, momentum) {
    'use strict';
    var result = new ENCOG.PropagationTrainer();
    result.network = network;
    result.trainingInput = input;
    result.trainingIdeal = ideal;
    result.type = type;
    result.learningRate = learningRate;
    result.momentum = momentum;

    result.layerDelta = ENCOG.ArrayUtil.newFloatArray(network.layerOutput.length);
    result.gradients = ENCOG.ArrayUtil.newFloatArray(network.weights.length);
    result.lastGradient = ENCOG.ArrayUtil.newFloatArray(network.weights.length);
    result.lastDelta = ENCOG.ArrayUtil.newFloatArray(network.weights.length);
    result.actual = ENCOG.ArrayUtil.newFloatArray(network.outputCount);
    result.flatSpot = ENCOG.ArrayUtil.newFloatArray(network.layerOutput.length);

    result.updateValues = ENCOG.ArrayUtil.newFloatArray(network.weights.length);

    ENCOG.ArrayUtil.fillArray(result.lastGradient, 0, result.lastGradient.length, 0);
    ENCOG.ArrayUtil.fillArray(result.updateValues, 0, result.updateValues.length, 0.1);
    ENCOG.ArrayUtil.fillArray(result.flatSpot, 0, network.weights.length, 0);

    return result;
};

ENCOG.Swarm = function () {
    'use strict';
};


ENCOG.Swarm.prototype = {
 
    NAME : 'Swarm',

   
    agents : null,

    
    callbackNeighbors : null,

    
    constCohesion : 0.01,

    
    constAlignment : 0.5,

    
    constSeparation : 0.25,

    iteration : function () {
        'use strict';
        var i, neighbors, meanX, meanY, dx, dy, targetAngle, nearest, separation, alignment, cohesion, turnAmount;

        
        for (i = 0; i < this.agents.length; i += 1) {
           
            targetAngle = 0;

            neighbors = ENCOG.MathUtil.kNearest(this.agents[i], this.agents, 5, Number.MAX_VALUE, 0, 2);
            nearest = ENCOG.MathUtil.kNearest(this.agents[i], this.agents, 5, 10, 0, 2);

           
            separation = 0;
            if (nearest.length > 0) {
                meanX = ENCOG.ArrayUtil.arrayMean(nearest, 0);
                meanY = ENCOG.ArrayUtil.arrayMean(nearest, 1);
                dx = meanX - this.agents[i][0];
                dy = meanY - this.agents[i][1];
                separation = (Math.atan2(dx, dy) * 180 / Math.PI) - this.agents[i][2];
                separation += 180;
            }

            
            alignment = 0;

            if (neighbors.length > 0) {
                alignment = ENCOG.ArrayUtil.arrayMean(neighbors, 2) - this.agents[i][2];
            }

            if (this.callbackNeighbors !== null) {
                this.callbackNeighbors(i, neighbors);
            }

            
            cohesion = 0;

            if (neighbors.length > 0) {
                meanX = ENCOG.ArrayUtil.arrayMean(this.agents, 0);
                meanY = ENCOG.ArrayUtil.arrayMean(this.agents, 1);
                dx = meanX - this.agents[i][0];
                dy = meanY - this.agents[i][1];
                cohesion = (Math.atan2(dx, dy) * 180 / Math.PI) - this.agents[i][2];
            }

          
            turnAmount = (cohesion * this.constCohesion) + (alignment * this.constAlignment) + (separation * this.constSeparation);

            this.agents[i][2] += turnAmount;

          
        }
    }
};

ENCOG.Swarm.create = function (agents) {
    'use strict';
    var result = new ENCOG.Swarm();
    result.agents = agents;
    return result;
};

ENCOG.Anneal = function () {
    'use strict';
};


ENCOG.Anneal.prototype = {
    
    NAME : 'Anneal',

   
    solution : null,

   
    scoreSolution : null,

    
    randomize : null,

    
    constStartTemp : 10.0,

    
    constStopTemp : 2.0,

    
    constCycles : 10,


    iteration : function () {
        'use strict';

        var bestArray, temperature, bestScore, curScore, i;

        bestArray = this.solution.slice();

        temperature = this.constStartTemp;
        bestScore = this.scoreSolution(this.solution);

        for (i = 0; i < this.constCycles; i += 1) {
            this.randomize(this.solution, temperature);
            curScore = this.scoreSolution(this.solution);

            if (curScore < bestScore) {
                bestArray = this.solution.slice();
                bestScore = curScore;
            }

            this.solution = bestArray.slice();

            temperature *= Math.exp(Math.log(this.constStopTemp
                / this.constStartTemp)
                / (this.constCycles - 1));
        }
    }
};

ENCOG.Anneal.create = function (solution) {
    'use strict';
    var result = new ENCOG.Anneal();
    result.solution = solution;
    return result;
};

ENCOG.Genetic = function () {
    'use strict';
};


ENCOG.Genetic.prototype = {
   
    NAME : 'Genetic',

    
    population : null,

   
    scoreSolution : null,

    
    mutate : null,

  
    crossover : null,

    
    constMutationPercent : 0.1,

    
    constMatePercent : 0.24,

    
    constMatingPopulationPercent : 0.5,

    iteration : function () {
        'use strict';

        var countToMate, offspringCount, offspringIndex, matingPopulationSize, motherID, fatherID;

        countToMate = Math.floor(this.population.length * this.constMatePercent);
        offspringCount = countToMate * 2;
        offspringIndex = this.population.length - offspringCount;
        matingPopulationSize = Math.floor(this.population.length * this.constMatingPopulationPercent);

       
        for (motherID = 0; motherID < countToMate; motherID++) {
            fatherID = Math.floor(Math.random() * matingPopulationSize);
            this.crossover(
                this.population[motherID].data,
                this.population[fatherID].data,
                this.population[offspringIndex].data,
                this.population[offspringIndex + 1].data);

            if (Math.random() > this.constMutationPercent) {
                this.mutate(this.population[offspringIndex].data);
            }

            if (Math.random() > this.constMutationPercent) {
                this.mutate(this.population[offspringIndex].data);
            }

           
            this.population[offspringIndex].score = this.scoreSolution(this.population[offspringIndex].data);
            this.population[offspringIndex + 1].score = this.scoreSolution(this.population[offspringIndex + 1].data);

           
            offspringIndex += 2;
        }

        this.sortPopulation();
    },
    createPopulation : function (size, generate) {
        'use strict';
        var i, d, l;

        this.population = [];
        for (i = 0; i < size; i++) {
            d = generate();
            l = this.scoreSolution(d);
            this.population[i] = {
                'data' : d,
                'score' : l
            };
        }

        this.sortPopulation();
    },
    getSolution : function () {
        return this.population[0].data;
    },
    sortPopulation : function () {
        this.population.sort(function (a, b) {
            return a.score - b.score
        });
    }
};


ENCOG.Genetic.create = function () {
    return new ENCOG.Genetic();
};


ENCOG.SOM = function () {
    'use strict';
};

ENCOG.SOM.prototype = {

   
    NAME : "SOM",

   
    weights : null,

   
    inputCount : 0,

   
    outputCount : 0,

   
    classify : function (inputData) {
        'use strict';

        var minDist, result, i, dist;

        if (inputData.length > this.inputCount) {
            throw new Error(
                "Can't classify SOM with input size of " + this.inputCount
                    + " with input data of count " + inputData.length);
        }

        minDist = Number.POSITIVE_INFINITY;
        result = -1;

        for (i = 0; i < this.outputCount; i += 1) {
            dist = ENCOG.MathUtil.euclideanDistance(inputData, this.weights[i], 0, this.inputCount);
            if (dist < minDist) {
                minDist = dist;
                result = i;
            }
        }

        return result;
    }
};

ENCOG.SOM.create = function (theInputCount, theOutputCount) {
    'use strict';
    var result = new ENCOG.SOM();
    result.inputCount = theInputCount;
    result.outputCount = theOutputCount;
    result.weights = ENCOG.ArrayUtil.allocateBoolean2D(theOutputCount, theInputCount);
    return result;
};

ENCOG.TrainSOM = function () {
    'use strict';
};

ENCOG.TrainSOM.prototype = {

    
    NAME : "SOM",

    weights : null,
    som : null,
    learningRate : 0.5,
    correctionMatrix : null,
    trainingInput : null,
    worstDistance : 0,


    
    iteration : function () {
        'use strict';

        var i, input, bmu;

      
        ENCOG.ArrayUtil.fillArray2D(this.correctionMatrix, 0);

       
        for (i = 0; i < this.trainingInput.length; i++) {
            input = this.trainingInput[i];

            bmu = this.calculateBMU(input);

            this.train(bmu, input);

            this.applyCorrection();
        }

      

    },
    reset : function () {
        ENCOG.MathUtil.randomizeArray2D(this.weights, -1, 1);
    },
    calculateBMU : function (input) {
        var result, lowestDistance, i, distance;

        result = 0;

        if (input.length > this.som.inputCount) {
            throw new Error(
                "Can't train SOM with input size of " + this.inputCount
                    + " with input data of count " + input.length);
        }

        lowestDistance = Number.POSITIVE_INFINITY;

        for (i = 0; i < this.som.outputCount; i++) {
            distance = ENCOG.MathUtil.euclideanDistance(this.som.weights[i], input, 0, this.som.weights[i].length);

          
            if (distance < lowestDistance) {
                lowestDistance = distance;
                result = i;
            }
        }

        
        if (lowestDistance > this.worstDistance) {
            this.worstDistance = lowestDistance;
        }

        return result;
    },
    train : function (bmu, input) {

    },
    applyCorrection : function () {

    }
};

ENCOG.TrainSOM.create = function (theSom, theLearningRate) {
    'use strict';
    var result = new ENCOG.TrainSOM();
    result.som = theSom;
    result.learningRate = theLearningRate;
    result.correctionMatrix = ENCOG.ArrayUtil.allocateBoolean2D(this.som.outputCount, this.som.inputCount);
    return result;
};

ENCOG.ReadCSV = function () {
    'use strict';
};

ENCOG.ReadCSV.prototype = {
  
    regStr : null,

   
    inputData : null,

    
    idealData : null,

   
    inputCount : 0,

   
    idealCount : 0,

   
    delimiter : ',',

    readCSV : function (csv, theInputCount, theIdealCount) {
        var currentIndex, regex, matches, value, d;

        this.inputCount = theInputCount;
        this.idealCount = theIdealCount;

        regex = new RegExp(this.regStr, "gi");

       
        this.inputData = [
            []
        ];
        this.idealData = [
            []
        ];

        currentIndex = 0;

        while (matches = regex.exec(csv)) {
          
            d = matches[ 1 ];

           
            if (d.length && (d != this.delimiter)) {
                this.inputData.push([]);
                this.idealData.push([]);
                currentIndex = 0;
            }

          
            if (matches[ 2 ]) {
                value = matches[ 2 ].replace(
                    new RegExp("\"\"", "g"),
                    "\""
                );

            } else {
                value = matches[ 3 ];
            }

         
            if (currentIndex < this.inputCount) {
                this.inputData[ this.inputData.length - 1 ].push(value);
            } else {
                this.idealData[ this.idealData.length - 1 ].push(value);
            }
            currentIndex += 1;
        }
    }
};

ENCOG.ReadCSV.create = function (theDelimiter) {
    'use strict';
    var result = new ENCOG.ReadCSV();

    result.delimiter = (theDelimiter || ",");

    result.regStr =
     
        "(\\" + result.delimiter + "|\\r?\\n|\\r|^)" +
           
            "(?:\"([^\"]*(?:\"\"[^\"]*)*)\"|" +
           
            "([^\"\\" + result.delimiter + "\\r\\n]*))";
    return result;
};

ENCOG.ReadCSV.fromCommaListInt = function (str) {
    'use strict';
    var result, parts, i;

    result = [];
    parts = str.split(',');

    for (i = 0; i < parts.length; i+=1) {
        result.push(parseInt(parts[i],10));
    }


    return result;
};

ENCOG.ReadCSV.fromCommaListFloat = function (str) {
    'use strict';
    var result, parts, i;

    result = [];
    parts = str.split(',');

    for (i = 0; i < parts.length; i+=1) {
        result.push(parseFloat(parts[i]));
    }

    return result;
};

ENCOG.ReadCSV.toCommaList = function (arr) {
    'use strict';
    var result, i;

    result = '';

    for (i = 0; i < arr.length; i+=1) {
        if (i > 0) {
            result += ',';
        }
        result += arr[i];
    }

    return result;
};

ENCOG.EGFILE = function () {
    'use strict';
};


ENCOG.EGFILE.save = function (obj) {
    'use strict';
    var result = "", now, i, af;

    now = (new Date()).getTime();

    result += 'encog,BasicNetwork,' + ENCOG.PLATFORM + ',3.1.0,1,' + now + ENCOG.NEWLINE;
    result += '[BASIC]' + ENCOG.NEWLINE;
    result += '[BASIC:PARAMS]' + ENCOG.NEWLINE;
    result += '[BASIC:NETWORK]' + ENCOG.NEWLINE;
    result += 'beginTraining=' + obj.beginTraining + ENCOG.NEWLINE;
    result += 'connectionLimit=' + obj.connectionLimit + ENCOG.NEWLINE;
    result += 'contextTargetOffset=' + ENCOG.ReadCSV.toCommaList(obj.contextTargetOffset) + ENCOG.NEWLINE;
    result += 'contextTargetSize=' + ENCOG.ReadCSV.toCommaList(obj.contextTargetSize) + ENCOG.NEWLINE;
    result += 'endTraining=' + obj.endTraining + ENCOG.NEWLINE;
    result += 'hasContext=' + (obj.hasContext ? 't' : 'f') + ENCOG.NEWLINE;
    result += 'inputCount=' + obj.inputCount + ENCOG.NEWLINE;
    result += 'layerCounts=' + ENCOG.ReadCSV.toCommaList(obj.layerCounts) + ENCOG.NEWLINE;
    result += 'layerFeedCounts=' + ENCOG.ReadCSV.toCommaList(obj.layerFeedCounts) + ENCOG.NEWLINE;
    result += 'layerContextCount=' + ENCOG.ReadCSV.toCommaList(obj.layerContextCount) + ENCOG.NEWLINE;
    result += 'layerIndex=' + ENCOG.ReadCSV.toCommaList(obj.layerIndex) + ENCOG.NEWLINE;
    result += 'output=' + ENCOG.ReadCSV.toCommaList(obj.layerOutput) + ENCOG.NEWLINE;
    result += 'outputCount=' + obj.outputCount + ENCOG.NEWLINE;
    result += 'weightIndex=' + ENCOG.ReadCSV.toCommaList(obj.weightIndex) + ENCOG.NEWLINE;
    result += 'weights=' + ENCOG.ReadCSV.toCommaList(obj.weights) + ENCOG.NEWLINE;
    result += 'biasActivation=' + ENCOG.ReadCSV.toCommaList(obj.biasActivation) + ENCOG.NEWLINE;
    result += '[BASIC:ACTIVATION]' + ENCOG.NEWLINE;
    for (i = 0; i < obj.activationFunctions.length; i+=1) {
        af = obj.activationFunctions[i];
        result += '\"';
        result += af.NAME;
        result += '\"' + ENCOG.NEWLINE;
    }
    return result;
};

ENCOG.EGFILE.load = function (str) {
    'use strict';
    var lines, currentLine, parts;

    currentLine = 0;

    lines = str.match(/^.*([\n\r]+|$)/gm);

    while (lines[currentLine].trim().length === 0) {
        currentLine+=1;
    }

    parts = lines[currentLine].trim().split(',');

    if (parts[0] !== 'encog') {
        throw new Error("Not a valid Encog EG file.");
    }

    if (parts[1] === 'BasicNetwork') {
        return ENCOG.EGFILE.loadBasicNetwork(str);
    } else {
        throw new Error("Encog Javascript does not support: " + parts[1]);
    }
};

ENCOG.EGFILE._loadNetwork = function (lines, currentLine, result) {
    var idx, line, name, value;

    while (currentLine < lines.length) {
        line = lines[currentLine].trim();

        if (line[0] == '[') {
            break;
        }

        currentLine++;
        idx = line.indexOf('=');

        if (idx == -1) {
            throw new Error("Invalid line in BasicNetwork file: " + line);
        }

        name = line.substr(0, idx).trim().toLowerCase();
        value = line.substr(idx + 1).trim();

        if (name == 'begintraining') {
            result.beginTraining = parseInt(value);
        }
        else if (name == 'connectionlimit') {
            result.connectionLimit = parseFloat(value);
        }
        else if (name == 'contexttargetoffset') {
            result.contextTargetOffset = ENCOG.ReadCSV.fromCommaListInt(value);
        }
        else if (name == 'contexttargetsize') {
            result.contextTargetSize = ENCOG.ReadCSV.fromCommaListInt(value);
        }
        else if (name == 'endtraining') {
            result.endTraining = parseInt(value);
        }
        else if (name == 'hascontext') {
            result.hasContext = (value.toLowerCase() == 'f');
        }
        else if (name == 'inputcount') {
            result.inputCount = parseInt(value);
        }
        else if (name == 'layercounts') {
            result.layerCounts = ENCOG.ReadCSV.fromCommaListInt(value);
        }
        else if (name == 'layerfeedcounts') {
            result.layerFeedCounts = ENCOG.ReadCSV.fromCommaListInt(value);
        }
        else if (name == 'layercontextcount') {
            result.layerContextCount = ENCOG.ReadCSV.fromCommaListInt(value);
        }
        else if (name == 'layerindex') {
            result.layerIndex = ENCOG.ReadCSV.fromCommaListInt(value);
        }
        else if (name == 'output') {
            result.layerOutput = ENCOG.ReadCSV.fromCommaListFloat(value);
        }
        else if (name == 'outputcount') {
            result.outputCount = parseInt(value);
        }
        else if (name == 'weightindex') {
            result.weightIndex = ENCOG.ReadCSV.fromCommaListInt(value);
        }
        else if (name == 'weights') {
            result.weights = ENCOG.ReadCSV.fromCommaListFloat(value);
        }
        else if (name == 'biasactivation') {
            result.biasActivation = ENCOG.ReadCSV.fromCommaListFloat(value);
        }
    }

    result.layerSums = [];
    ENCOG.ArrayUtil.fillArray(result.layerSums, 0, result.layerSums, 0);

    return currentLine;
};

ENCOG.EGFILE._loadActivation = function (lines, currentLine, result) {
    var i, line;

    result.activationFunctions = [];

    i = 0;
    while (currentLine < lines.length) {

        line = lines[currentLine++].trim();

        if (line[0] == '[') {
            break;
        }

        line = ENCOG.Util.stripQuotes(line);

        if (line == 'ActivationLinear') {
            result.activationFunctions[i] = ENCOG.ActivationLinear.create();
        } else if (line == 'ActivationSigmoid') {
            result.activationFunctions[i] = ENCOG.ActivationSigmoid.create();
        } else if (line == 'ActivationTANH') {
            result.activationFunctions[i] = ENCOG.ActivationTANH.create();
        } else if (line == 'ActivationElliott') {
            result.activationFunctions[i] = ENCOG.ActivationElliott.create();
        } else if (line == 'ActivationElliottSymmetric') {
            result.activationFunctions[i] = ENCOG.ActivationElliottSymmetric.create();
        }

        i += 1;
    }

    return currentLine;
};

ENCOG.EGFILE.loadBasicNetwork = function (str) {
    var lines, currentLine, line, parts, result;

    currentLine = 0;

    lines = str.match(/^.*([\n\r]+|$)/gm);

    while (lines[currentLine].trim().length == 0) {
        currentLine++;
    }

    parts = lines[currentLine++].trim().split(',');

    if (parts[0] != 'encog') {
        throw new Error("Not a valid Encog EG file.");
    }

    if (parts[1] != 'BasicNetwork') {
        throw new Error("Not a BasicNetwork EG file.");
    }

    result = new ENCOG.BasicNetwork();

    while (currentLine < lines.length) {
        line = lines[currentLine++].trim();

        if (line == '[BASIC:NETWORK]') {
            currentLine = this._loadNetwork(lines, currentLine, result);
        } else if (line == '[BASIC:ACTIVATION]') {
            currentLine = this._loadActivation(lines, currentLine, result);
        }
    }

    return result;
};
