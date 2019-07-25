var StockPredictor = function () {
	this.network = null;
	this.TRAIN_EPOCH = 5000;
	this.MAX_LEN = 50;
	
	this.dataList = [];
	this.DATA = [];
	
	this.data = null;
	
	this.init();
}

StockPredictor.prototype = {
	init: function () {
		// DATA 설정
		for (let i = 97; i <= 122; i++) // 'a' ~ 'z'
			this.DATA.push(String.fromCharCode(i));
		this.DATA.push(' ');
		this.DATA.push(null);
		
		// hell -> ello
		this.dataList = [
			"hello mister",
			"host bar",
			"hi my name is hyeon",
			"kim kang won",
			"cold weather has a great effect on how our minds and our bodies work"
		];
		
		this.layerNum = {input: 4, hidden: 12, output: 4};
		this.layerSize = {input: this.DATA.length, hidden: this.DATA.length, output: this.DATA.length};
		this.network = new LSTM(this.layerNum, this.layerSize);
	},
	
	predict: function (data) {
		if (data.length < this.layerNum.input)
			return null;
		
		this.data = data;
		let inputList = this.encode(data);
		let input = [];
		
		inputList.pop(); // null 삭제
		for (let i = 0; i < this.layerNum.input; i++)
			input.push(inputList.shift());
		
		// 입력되는 값
		this.network.initStack();
		while (inputList.length > 0) {
			// console.log('input: ' + this.decode(input));
			this.network.predict(input);
			
			input.push(inputList.shift());
			input.shift();
		}
		
		// 추측되는 값
		let predictStr = data;
		for (let i = 0; i < this.MAX_LEN; i++) {
			// console.log('input: ' + this.decode(input));
			let prob = this.network.predict(input);
			prob = prob.pop();
			let char = this.decode([prob]);
			// console.log('input: ' + this.decode(input) + ' | \'' + char + '\'');
			
			if (char == "") // null을 의미
				break;
			predictStr += char;
			
			input.push(this.encode(char)[0]); // prob는 확률이기 때문에 사용하지 않음
			input.shift();
		}
		
		return predictStr;
	},
	train: function () {
		for (let i = 0; i < this.TRAIN_EPOCH; i++) {
			for (let j = 0; j < this.dataList.length; j++) {
				let data = this.encode(this.dataList[j]);
				let temp = 0;
				
				this.network.initStack();
				temp = data.length - ((this.layerNum.input < this.layerNum.output) ? this.layerNum.input : this.layerNum.output);
				for (let k = 0; k < temp; k++) {
					let input = data.slice(k, k+this.layerNum.input);
					let label = data.slice(k+1, k+1+this.layerNum.output);
					// console.log('input: ' + this.decode(input) + ' label: ' + this.decode(label));
					this.network.train(input, label);
				}
			}
			console.log('epoch: ' + i);
		}
	},
	
	evaluate: function () {
		let res;
		let input, output, label;
		
		input = 'hell';
		output = this.predict(input);
		label = this.dataList[0];
		document.body.append("input: " + input + " output: " + output + " label: " + label);
		
		document.body.appendChild(document.createElement('br'));
		
		input = 'host';
		output = this.predict(input);
		label = this.dataList[1];
		document.body.append("input: " + input + " output: " + output + " label: " + label);
	},
	
	encode: function (str) {
		let dataList = [];
		
		for (let i = 0; i < str.length; i++) {
			let data = new Array(this.DATA.length).fill(0);
			data[this.DATA.indexOf(str[i])] = 1;
			dataList.push(data);
		}
		
		// null
		let data = new Array(this.DATA.length).fill(0);
		data[this.DATA.indexOf(null)] = 1;
		dataList.push(data);
		
		return dataList;
	},
	decode: function (dataList) {
		let str = "";
		
		for (let i = 0; i < dataList.length; i++) {
			let data = dataList[i];
			let index = 0;
			
			for (let i = 1; i < data.length; i++) {
				if (data[i] > data[index])
					index = i;
			}
			
			if (this.DATA[index] == null)
				break;
			str += this.DATA[index];
		}
		
		return str;
	},
	
	getData: function () {
		return this.data;
	}
}

var stockPredictor = new StockPredictor();
stockPredictor.train();
stockPredictor.evaluate();
