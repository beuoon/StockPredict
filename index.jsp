<%@ page contentType = "text/html;charset=utf-8" %>

<html>
	<head>
		<script src="//code.jquery.com/jquery-3.4.1.min.js"></script>
	</head>
	<body>
		<input id="inputText" type="text" onChange="console.log()"/><br>
		예측값: <text id="resultText"></text><br>
		
        <script src="LSTM/Util.js"></script>
		<script src="LSTM/Layer_Output.js"></script>
        <script src="LSTM/Layer_Hidden.js"></script>
        <script src="LSTM/LSTM.js"></script>
		
		<!--
		<script src="RNN/Layer_Output.js"></script>
        <script src="RNN/Layer_Hidden.js"></script>
        <script src="RNN/RNN.js"></script>
		-->
		
		<script src="StockPredictor.js"></script>
		
		<script type="text/javascript">
			let inputValue = null;
			$("#inputText").on("propertychange change keyup paste input", function() {
				let value = $(this).val();
				if (stockPredictor.getData() == value)
					return ;
				let output = stockPredictor.predict(value);
				if (output != null)
					$("#resultText").text(output);
			});
		</script>
	</body>
</html>