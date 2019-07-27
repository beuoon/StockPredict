/*
	RNN과 LSTM에 대한 이해: https://ratsgo.github.io/natural%20language%20processing/2017/03/09/rnnlstm/
*/
let RNN = function (layerNum, layerSize) {
	this.inputNum = layerNum.input, this.hiddenNum = layerNum.hidden, this.outputNum = layerNum.output;
	this.inputSize = layerSize.input, this.hiddenSize = layerSize.hidden, this.outputSize = layerSize.output;
	
	this.hiddenLayer = [];
	this.outputLayer = [];
	for (let i = 0; i < this.hiddenNum; i++)
		this.hiddenLayer[i] = new Layer_Hidden(this.inputSize, this.hiddenSize);
	for (let i = 0; i < this.outputNum; i++)
		this.outputLayer[i] = new Layer_Output(this.hiddenSize, this.outputSize);
	
	this.BLANK_X = new Array(this.inputSize).fill(0); // 입력 레이어가 없는 은닉 레이어에서 사용
	this.initStack();
}
RNN.prototype = {
	predict: function (inputList) {
		return this.forward(inputList);
	},
	train: function (inputList, labelList) {
		let outputList = this.forward(inputList);
		
		let deltaList = [];
		for (let i = 0; i < this.outputNum; i++) {
			deltaList[i] = [];
			for (let j = 0; j < this.outputSize; j++)
				deltaList[i][j] = outputList[i][j] - labelList[i][j];
		}
		
		this.backward(deltaList);
	},
	
	initStack: function () {
		this.hStack = new Array(this.hiddenSize).fill(0);
	},
	
	forward: function (inputList) {
		let prevH = this.hStack.slice();
		
		// Hidden Layer
		for (let i = 0; i < this.hiddenNum; i++) {
			if (i < this.inputNum)
				prevH = this.hiddenLayer[i].forward(prevH, inputList[i]);
			else
				prevH = this.hiddenLayer[i].forward(prevH, this.BLANK_X);
			
			if (i == 0)
				this.hStack = prevH.slice();
		}
		
		// Output Layer
		let outputList = [];
		for (let i = 0, j = this.hiddenNum - this.outputNum; i < this.outputNum; i++, j++) {
			let h = this.hiddenLayer[j].getH();
			outputList[i] = this.outputLayer[i].forward(h);
		}
		
		return outputList;
	},
	backward: function (deltaList) {
		let dh = new Array(this.hiddenSize).fill(0);
		
		// Hidden Layer 
		for (let i = this.hiddenNum-1, j = this.outputNum-1; i >= 0; i--, j--) {
			// Output Layer
			if (j >= 0) {
				let dy = this.outputLayer[j].backward(deltaList[j]);
				for (let k = 0; k < this.hiddenSize; k++)
					dh[k] += dy[k];
			}
			dh = this.hiddenLayer[i].backward(dh);
		}
	}
}