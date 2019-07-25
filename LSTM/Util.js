
function add(arr1, arr2) {
	let arr = [];
	
	for (let i = 0; i < arr1.length; i++)
		arr[i] = arr1[i] + arr2[i];
	
	return arr;
}
function mul(arr1, arr2) {
	let arr = [];
	
	for (let i = 0; i < arr1.length; i++)
		arr[i] = arr1[i] * arr2[i];
	
	return arr;
}

function sigmoid(x) {
	let arr = [];
	
	for (let i = 0; i < x.length; i++)
		arr[i] = 1 / (1 + Math.exp(-x[i]));
	
	return arr;
}
function sigmoid_diff(x) {
	function f (x) {
        return 1 / (1 + Math.exp(-x));
    }
    
    let arr = [];
	
	for (let i = 0; i < x.length; i++) {
		let fx = f(x[i]);
		arr[i] = fx * (1 - fx);
	}
	
	return arr;
}


function tanh(x) {
	let arr = [];
	
	for (let i = 0; i < x.length; i++)
		arr[i] = Math.tanh(x[i]);
	
	return arr;
}
function tanh_diff(x) {
	let arr = [];
	
	for (let i = 0; i < x.length; i++)
		arr[i] = 1 - Math.pow(Math.tanh(x[i]), 2);
	
	return arr;
}