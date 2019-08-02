let Layer_Output = function (hSize, ySize, softmax = false) {
    this.ETA = 0.005;
    this.hSize = hSize, this.ySize = ySize; // 입력 크기, 출력 크기
    this.weight = new Array(this.ySize);
	this.weightMoment = new Array(this.ySize);
    this.h;
	
	this.softmax = softmax;
        
    let weightLimit = Math.sqrt(6.0/this.hSize); // He 초기화 사용 변수
    for (let i = 0; i < this.ySize; i++) {
        this.weight[i] = [];
        for (let j = 0; j <= this.hSize; j++)
            this.weight[i][j] = (Math.random() * 2 - 1) * weightLimit;
		
		this.weightMoment[i] = new Array(this.hSize+1).fill(0);
    }
}

Layer_Output.prototype = {
    forward: function (h) {
		let y = new Array(this.ySize).fill(0);
		this.h = h.slice();
		
		for (let i = 0; i < this.ySize; i++) {
			for (let j = 0; j < this.hSize; j++)
				y[i] += this.h[j] * this.weight[i][j];
			y[i] += this.weight[i][this.hSize]; // Bias
		}
		
		if (this.softmax) {
			// TODO: Softmax를 구현하자
		}
		
		return y;
    },
    backward: function (delta) {
		let dh = new Array(this.hSize).fill(0);
        
		for (let i = 0; i < this.hSize; i++) {
			for (let j = 0; j < this.ySize; j++)
				dh[i] += delta[j] * this.weight[j][i];
        }
        
		for (let i = 0; i < this.ySize; i++) {
			for (let j = 0; j < this.hSize; j++) {
				this.weightMoment[i][j] = 0.9*this.weightMoment[i][j] + 0.1*(delta[i] * this.h[j]);
				this.weight[i][j] += -this.ETA * this.weightMoment[i][j];
			}
			
			// bias
			this.weightMoment[i][this.hSize] = 0.9*this.weightMoment[i][this.hSize] + 0.1*delta[i];
			this.weight[i][this.hSize] += -this.ETA * this.weightMoment[i][this.hSize];
		}
		
        return dh;
    }
}