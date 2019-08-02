let Layer_Hidden = function (xSize, hSize) {
    this.ETA = 0.005;
	
	this.hSize = hSize, this.xSize = xSize;
    this.weight_hh = [], this.weight_xh = [];
	this.weight_hh_moment = [], this.weight_xh_moment = [];
	this.bias = [];
	this.bais_moment = new Array(this.hSize).fill(0);
    this.prevH = null, this.x = null, this.h = null;
    
    let hhWeightLimit = Math.sqrt(6.0/this.hSize);	// He 초기화 사용 변수
    let xhWeightLimit = Math.sqrt(6.0/this.xSize);	// He 초기화 사용 변수
	let biasLimit = Math.sqrt(6.0/(this.hSize + this.xSize));
    for (let i = 0; i < this.hSize; i++) {
        this.weight_hh[i] = [];
		this.weight_xh[i] = [];
        
        for (let j = 0; j < this.hSize; j++)
            this.weight_hh[i][j] = (Math.random() * 2 - 1) * hhWeightLimit;
		for (let j = 0; j < this.xSize; j++)
			this.weight_xh[i][j] = (Math.random() * 2 - 1) * xhWeightLimit;
		this.bias[i] = (Math.random() * 2 - 1) * biasLimit;
		
		this.weight_hh_moment[i] = new Array(this.hSize).fill(0);
		this.weight_xh_moment[i] = new Array(this.xSize).fill(0);
    }
}
    
Layer_Hidden.prototype = {
    forward: function (prevH, x) {
		let h = new Array(this.hSize).fill(0);
		this.prevH = prevH.slice();
		this.x = x.slice();
		
		for (let i = 0; i < this.hSize; i++) {
			for (let j = 0; j < this.hSize; j++)
				h[i] += this.prevH[j] * this.weight_hh[i][j];
			for (let j = 0; j < this.xSize; j++)
				h[i] += this.x[j] * this.weight_xh[i][j];
			h[i] += this.bias[i];
			
			h[i] = Math.tanh(h[i]);
		}
		this.h = h.slice();
		
		return h;
    },
    backward: function (delta) {
		let dh = new Array(this.hSize).fill(0);
		for (let i = 0; i < this.hSize; i++)
			dh[i] = (1 - Math.pow(this.h[i], 2)) * delta[i];
		
		// backward delta
		let dh_prev = new Array(this.hSize).fill(0);
		for (let i = 0; i < this.hSize; i++) {
			for (let j = 0; j < this.hSize; j++)
				dh_prev[i] += dh[j] * this.weight_hh[j][i];
		}
			
		// update Weight
		for (let i = 0; i < this.hSize; i++) {
			for (let j = 0; j < this.hSize; j++) {
				this.weight_hh_moment[i][j] = 0.9*this.weight_hh_moment[i][j] + 0.1*(dh[i] * this.prevH[j]);
				this.weight_hh[i][j] += -this.ETA * this.weight_hh_moment[i][j];
			}
			
			for (let j = 0; j < this.xSize; j++) {
				this.weight_xh_moment[i][j] = 0.9*this.weight_xh_moment[i][j] + 0.1*(dh[i] * this.x[j]);
				this.weight_xh[i][j] += -this.ETA * this.weight_xh_moment[i][j];
			}
			
			this.bias_moment[i] = 0.9*this.bias_moment[i] + 0.1*dh[i];
			this.bias[i] += -this.ETA * this.bias_moment[i];
		}
		
        
        return dh_prev;
    },
	
	getH: function () {
		return this.h;
	}
}