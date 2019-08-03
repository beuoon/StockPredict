/*
	RNN과 LSTM에 대한 이해: https://ratsgo.github.io/natural%20language%20processing/2017/03/09/rnnlstm/
*/
let LSTM = function (layerNum, layerSize) {
	this.inputNum = layerNum.input, this.hiddenNum = layerNum.hidden, this.outputNum = layerNum.output;
	this.inputSize = layerSize.input, this.hiddenSize = layerSize.hidden, this.outputSize = layerSize.output;
	
	this.hiddenLayer = [];
	this.outputLayer = [];
	for (let i = 0; i < this.hiddenNum; i++)
		this.hiddenLayer[i] = new Layer_Hidden(this.inputSize, this.hiddenSize);
	for (let i = 0; i < this.outputNum; i++)
		this.outputLayer[i] = new Layer_Output(this.hiddenSize, this.outputSize);
	
	this.BLANK_X = new Array(this.inputSize).fill(0); // 입력 레이어가 없는 은닉 레이어에서 사용
	this.BLANK_DY = new Array(this.hiddenSize).fill(0); // 입력 레이어가 없는 은닉 레이어에서 사용
	this.initStack();
}
LSTM.prototype = {
	predict: function (inputList) {
		return this.forward(inputList);
	},
	train: function (inputList, labelList) {
		let error = 0;
		let outputList = this.forward(inputList);
		
		let deltaList = [];
		for (let i = 0; i < this.outputNum; i++) {
			deltaList[i] = [];
			for (let j = 0; j < this.outputSize; j++) {
				deltaList[i][j] = outputList[i][j] - labelList[i][j];
				error += 0.5*Math.pow(deltaList[i][j], 2);
			}
		}
		// error /= this.outputNum * this.outputSize;
		
		this.backward(deltaList);
		
		return error;
	},
	
	initStack: function () {
		this.h_stack = new Array(this.hiddenSize).fill(0);
		this.c_stack = new Array(this.hiddenSize).fill(0);
	},
	
	forward: function (inputList) {
		// this.initStack();
		let h_prev = this.h_stack;
		let c_prev = this.c_stack;
		
		// Hidden Layer
		for (let i = 0; i < this.hiddenNum; i++) {
			let x = (i < this.inputNum) ? inputList[i] : this.BLANK_X;
			let res = this.hiddenLayer[i].forward(x, h_prev, c_prev);
			// console.log('h: ' + res.h + ' c: ' + res.c);
			
			h_prev = res.h;
			c_prev = res.c;
			
			if (i == 0) {
				this.h_stack = h_prev;
				this.c_stack = c_prev;
			}
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
		let dh_next = new Array(this.hiddenSize).fill(0);
		let dc_next = new Array(this.hiddenSize).fill(0);
		
		// Hidden Layer 
		for (let i = this.hiddenNum-1, j = this.outputNum-1; i >= 0; i--, j--) {
			let dy = (j >= 0) ? this.outputLayer[j].backward(deltaList[j]) : this.BLANK_DY;
			
			let res = this.hiddenLayer[i].backward(dy, dh_next, dc_next);
			dh_next = res.dh;
			dc_next = res.dc;
		}
	}
}