package com.example.tensorflowdemo

import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Rect
import android.graphics.YuvImage
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import com.example.tensorflowdemo.databinding.ActivityMainBinding
import com.google.common.util.concurrent.ListenableFuture
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import java.io.ByteArrayOutputStream


class MainActivity : AppCompatActivity() {
    lateinit var interpreter:Interpreter;
    var inputImageWidth:Int= 0;
    var inputImageHeight:Int=0;
    var modelInputSize: Int = 0 // will be inferred from TF Lite model.
    private val FLOAT_TYPE_SIZE = 4
    private val PIXEL_SIZE = 1
    lateinit var binding:ActivityMainBinding;
    private val REQUEST_CODE_PERMISSIONS = 101
    private val REQUIRED_PERMISSIONS = arrayOf("android.permission.CAMERA")
    private lateinit var cameraProviderFuture : ListenableFuture<ProcessCameraProvider>
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
         binding= ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        val context:Context=this
        val model= FileUtil.loadMappedFile(context, "tflitemodel.tflite")
        val interpreter= Interpreter(model)
        val inputShape = interpreter.getInputTensor(0).shape()
        this.inputImageWidth = inputShape[1]
        this.inputImageHeight = inputShape[2]
        Log.i("width",(this.inputImageWidth).toString())
        Log.i("height",this.inputImageWidth.toString())
        val dtype = interpreter.getInputTensor(0).dataType()
        Log.i("inputDtype", dtype.toString())
        modelInputSize = FLOAT_TYPE_SIZE * inputImageWidth *
                inputImageHeight * PIXEL_SIZE
        this.interpreter=interpreter;
        cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        if (allPermissionsGranted()) {
                cameraProviderFuture.addListener(Runnable {
                    val cameraProvider = cameraProviderFuture.get()
                    bindPreview(cameraProvider)
                }, ContextCompat.getMainExecutor(this))
        } else {
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS);
        }
    }
    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)

        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                cameraProviderFuture.addListener(Runnable {
                    val cameraProvider = cameraProviderFuture.get()
                    bindPreview(cameraProvider)
                }, ContextCompat.getMainExecutor(this))
            } else {
                Toast.makeText(this, "Permissions not granted by the user.", Toast.LENGTH_SHORT)
                    .show()
                finish()
            }
        }
    }

    private fun allPermissionsGranted(): Boolean {

        for (permission in REQUIRED_PERMISSIONS) {
            if (ContextCompat.checkSelfPermission(
                    this,
                    permission
                ) != PackageManager.PERMISSION_GRANTED
            ) {
                return false
            }
        }
        return true
    }
    fun bindPreview(cameraProvider : ProcessCameraProvider) {
        var preview : Preview = Preview.Builder()
            .build()
        var cameraSelector : CameraSelector = CameraSelector.Builder()
            .requireLensFacing(CameraSelector.LENS_FACING_BACK)
            .build()

        preview.setSurfaceProvider(binding.previewView.getSurfaceProvider())

        val imageAnalysis = ImageAnalysis.Builder()
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_BLOCK_PRODUCER)
            .build()

        imageAnalysis.setAnalyzer(ContextCompat.getMainExecutor(this), { imageProxy ->
            val bitmap = imageProxyToBitmap(imageProxy)
            if (bitmap != null) {
                val label = classify(bitmap)
                runOnUiThread {
                    binding.textView.text = label
                }
            }
            imageProxy.close()
        })

        var camera = cameraProvider.bindToLifecycle(this as LifecycleOwner, cameraSelector, preview,imageAnalysis)
    }

    private fun imageProxyToBitmap(imageProxy: androidx.camera.core.ImageProxy): Bitmap? {
        val planes = imageProxy!!.planes
        val yBuffer = planes[0].buffer
        val uBuffer = planes[1].buffer
        val vBuffer = planes[2].buffer
        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()
        val nv21 = ByteArray(ySize + uSize + vSize)
        yBuffer[nv21, 0, ySize]
        vBuffer[nv21, ySize, vSize]
        uBuffer[nv21, ySize + vSize, uSize]
        val yuvImage = YuvImage(nv21, ImageFormat.NV21, imageProxy.width, imageProxy.height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, yuvImage.width, yuvImage.height), 75, out)
        val imageBytes = out.toByteArray()
        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)    }
    override fun onDestroy(){
        super.onDestroy()
        this.interpreter.close()
    }

    private fun classify(bitmap: Bitmap): String {

        // TODO: Add code to run inference with TF Lite.
        // Pre-processing: resize the input image to match the model input shape.
        val imageProcessor =
            ImageProcessor.Builder()
                .add(ResizeOp(inputImageWidth, inputImageHeight, ResizeOp.ResizeMethod.BILINEAR))
                .build()
        //binding.imageView.setImageBitmap(bitmap)
        val myTensorImage= TensorImage(DataType.FLOAT32)
        val rgbBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        myTensorImage.load(rgbBitmap)
        val normaliser=ImageProcessor.Builder()
            .add(NormalizeOp(0f,255f))
            .build()

        var myimg=imageProcessor.process(myTensorImage)
        binding.imageView2.setImageBitmap(myimg.bitmap)
        myimg=normaliser.process(myimg)
        // Define an array to store the model output.
        val output = Array(1) { FloatArray(3) }
        Log.i("width",myimg.width.toString())
        Log.i("height",myimg.height.toString())
        Log.i("colorSpace",myimg.getColorSpaceType().toString())

        interpreter?.run(myimg.getTensorBuffer().buffer, output)

        // Post-processing: find the digit that has the highest probability
        // and return it a human-readable string.
        val result = output[0]
        Log.i("output", output[0].toString())
        Log.i("output", output[0][0].toString())
        Log.i("output", output[0][1].toString())
        Log.i("output", output[0][2].toString())
        val maxIndex = result.indices.maxByOrNull { result[it] } ?: -1
        val classes=arrayOf("Broccoli","Cauliflower","Unknown")
        val resultString =
            "Prediction Result: %s\nBroccoli: %2f\nCauliflower: %2f\nUnknown: %2f"
                .format(classes[maxIndex], result[0],result[1],result[2])

        return resultString
    }


}