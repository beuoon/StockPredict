var StockPredictor = function () {
	this.network = null;
	this.TRAIN_EPOCH = 20000;
	
	this.dataList = [];
	this.DATA_SIZE = 1;
	
	this.data = null;
	
	this.init();
}

StockPredictor.prototype = {
	init: function () {
		// close value
		this.dataList = [
			[[1901.75], [1902.25], [1950.63], [1938.4301], [1926.52], [1911.52], [1900.8199], [1962.46], [1950.55], [1921.], [1917.77], [1899.87], [1889.98], [1822.6801], [1840.12], [1871.15], [1907.5699], [1869.], [1858.97], [1857.52], [1859.6801], [1815.48], [1823.28], [1836.4301], [1819.1899], [1816.3199], [1775.0699], [1692.6899], [1729.5601], [1738.5], [1754.36], [1804.03], [1860.63], [1863.7], [1855.3199], [1870.3], [1869.67], [1886.03], [1901.37], [1908.79], [1918.1899], [1911.3], [1913.9], [1878.27], [1897.83], [1904.28], [1893.63], [1922.1899], [1934.3101], [1939.], [1942.91], [1952.3199], [1988.3], [2017.41], [2001.0699], [2011.], [2020.99], [2009.9], [1992.03], [1977.9], [1964.52], [1985.63], [1994.49], [2000.8101]]
		];
		this.data_u = [];
		this.data_s = [];
		
		for (let i = 0; i < this.dataList.length; i++) {
			// 평균 구하기
			let u = new Array(this.DATA_SIZE).fill(0);
			
			for (let j = 0; j < this.dataList[i].length; j++) {
				for (let k = 0; k < this.DATA_SIZE; k++)
					u[k] += this.dataList[i][j][k];
			}
			
			for (let k = 0; k < this.DATA_SIZE; k++)
				u[k] /= this.dataList[i].length;
			
			// 분산 구하기
			let v = new Array(this.DATA_SIZE).fill(0);
			
			for (let j = 0; j < this.dataList[i].length; j++) {
				for (let k = 0; k < this.DATA_SIZE; k++)
					v[k] += Math.pow(this.dataList[i][j][k] - u[k], 2);
			}
			
			for (let k = 0; k < this.DATA_SIZE; k++)
				v[k] /= this.dataList[i].length;
			
			// 표준화
			let s = [];
			for (let k = 0; k < this.DATA_SIZE; k++)
				s[k] = Math.sqrt(v[k]);
			
			for (let j = 0; j < this.dataList[i].length; j++) {
				for (let k = 0; k < this.DATA_SIZE; k++)
					this.dataList[i][j][k] = (this.dataList[i][j][k] - u[k]) / s[k];
			}
			
			this.data_u.push(u);
			this.data_s.push(s);
		}
		console.log(this.dataList);
		
		this.layerNum = {input: 3, hidden: 20, output: 3};
		this.layerSize = {input: this.DATA_SIZE, hidden: 5, output: this.DATA_SIZE};
		this.network = new RNN(this.layerNum, this.layerSize);
	},
	
	predict: function (data) {
		if (data.length < this.layerNum.input)
			return null;
		
		this.data = data;
		let inputList = data.slice();
		let input = [];
		
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
		// console.log(input);
		let output = this.network.predict(input);
		return output;
	},
	train: function () {
		for (let i = 0; i < this.TRAIN_EPOCH; i++) {
			for (let j = 0; j < this.dataList.length; j++) {
				let data = this.dataList[j];
				let temp = 0;
				
				this.network.initStack();
				temp = data.length -this.layerNum.input - this.layerNum.output;
				for (let k = 0; k < temp; k++) {
					let input = data.slice(k, k+this.layerNum.input);
					let label = data.slice(k+this.layerNum.input, k+this.layerNum.input+this.layerNum.output);
					// console.log('input: ' + input + ' label: ' + label);
					this.network.train(input, label);
				}
			}
			console.log('epoch: ' + i);
		}
	},
	
	evaluate: function () {
		let res;
		let input, output, label;
		
		let startIndex = 0;
		input = this.dataList[0].slice(startIndex, startIndex+this.layerNum.input);
		output = this.predict(input);
		label = this.dataList[0].slice(startIndex+this.layerNum.input, startIndex+this.layerNum.input+this.layerNum.output);
		document.body.append("output: ");
		
		for (let i = 0; i < output.length; i++) {
			let res = output[i][0] * this.data_s[0][0] + this.data_u[0][0];
			document.body.append(res + " ");
		}
		document.body.appendChild(document.createElement('br'));
		document.body.append("label: ");
		for (let i = 0; i < label.length; i++) {
			let res = label[i][0] * this.data_s[0][0] + this.data_u[0][0];
			document.body.append(res + " ");
		}
		document.body.appendChild(document.createElement('br'));
		
		document.body.appendChild(document.createElement('br'));
		
		
		startIndex = 20;
		input = this.dataList[0].slice(startIndex, startIndex+this.layerNum.input);
		output = this.predict(input);
		label = this.dataList[0].slice(startIndex+this.layerNum.input, startIndex+this.layerNum.input+this.layerNum.output);
		document.body.append("output: ");
		for (let i = 0; i < output.length; i++) {
			let res = output[i][0] * this.data_s[0][0] + this.data_u[0][0];
			document.body.append(res + " ");
		}
		document.body.appendChild(document.createElement('br'));
		document.body.append("label: ");
		for (let i = 0; i < label.length; i++) {
			let res = label[i][0] * this.data_s[0][0] + this.data_u[0][0];
			document.body.append(res + " ");
		}
	},
	
	getData: function () {
		return this.data;
	}
}

var stockPredictor = new StockPredictor();
stockPredictor.train();
stockPredictor.evaluate();
