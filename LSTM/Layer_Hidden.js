let Layer_Hidden = function (xSize, hSize) {
    this.ETA = 0.005;
	
	this.hSize = hSize, this.xSize = xSize;
    this.w_xh = [], this.w_hh = [];
	this.bias = [];
	
    this.x = null, this.h_prev = null, this.c_prev = null;
    this.f = null, this.i = null, this.g = null, this.o = null;
	this.h = null, this.c = null;
    
    let xhWeightLimit = Math.sqrt(6.0/this.xSize);	// He 초기화 사용 변수
    let hhWeightLimit = Math.sqrt(6.0/this.hSize);	// He 초기화 사용 변수
	let biasLimit = Math.sqrt(6.0/(this.hSize + this.xSize));
    for (let i = 0; i < this.hSize; i++) {
		this.w_xh[i] = [];
        this.w_hh[i] = [];
        
		for (let j = 0; j < this.xSize; j++)
			this.w_xh[i][j] = (Math.random() * 2 - 1) * xhWeightLimit;
        for (let j = 0; j < this.hSize; j++)
            this.w_hh[i][j] = (Math.random() * 2 - 1) * hhWeightLimit;
		this.bias[i] = (Math.random() * 2 - 1) * biasLimit;
    }
}
    
Layer_Hidden.prototype = {
    forward: function (x, h_prev, c_prev) {
		this.x = x.slice();
		this.h_prev = h_prev.slice();
		this.c_prev = c_prev.slice();
		
		let t = new Array(this.hSize).fill(0);
		for (let i = 0; i < this.hSize; i++) {
			for (let j = 0; j < this.xSize; j++)
				t[i] += x[j] * this.w_xh[i][j];
			for (let j = 0; j < this.hSize; j++)
				t[i] += h_prev[j] * this.w_hh[i][j];
			t[i] += this.bias[i];
		}
		
		this.f = sigmoid(t);
		this.i = sigmoid(t);
		this.g = tanh(t);
		this.o = sigmoid(t);
		
		this.c = add(mul(c_prev, this.f), mul(this.i, this.g));
		this.h = mul(tanh(this.c), this.o);
		
		return {h: this.h, c: this.c};
    },
    backward: function (dy, dh_next, dc_next) {
		let dh = add(dy, dh_next);
		
		let dc = add(mul(tanh_diff(this.c), mul(this.o, dh)), dc_next);
		let dc_prev = mul(dc, this.f);
		
		let df_ = mul(sigmoid_diff(this.f), mul(dc, this.c_prev));
		let di_ = mul(sigmoid_diff(this.i), mul(dc, this.g));
		let dg_ = mul(tanh_diff(this.g), mul(dc, this.i));
		let do_ = mul(sigmoid_diff(this.o), mul(dh, tanh(this.c)));
		
		let dt = add(add(df_, di_), add(dg_, do_));
		
		let dh_prev = new Array(this.hSize).fill(0);
		for (let i = 0; i < this.hSize; i++) {
			for (let j = 0; j < this.hSize; j++)
				dh_prev[i] += dt[j] * this.w_hh[j][i];
		}
		
		// update Weight
		for (let i = 0; i < this.hSize; i++) {
			for (let j = 0; j < this.xSize; j++)
				this.w_xh[i][j] += -this.ETA * dt[i] * this.x[j];
			for (let j = 0; j < this.hSize; j++)
				this.w_hh[i][j] += -this.ETA * dt[i] * this.h_prev[j];
			this.bias[i] += -this.ETA * dt[i];
		}
		
        return {dh: dh_prev, dc: dc_prev};
    },
	
	getH: function () {
		return this.h;
	}
}