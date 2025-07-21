package com.example.tensorflowdemo

import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.view.View
import android.widget.AdapterView
import android.widget.AdapterView.OnItemSelectedListener
import android.widget.ArrayAdapter
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
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp
import org.tensorflow.lite.support.image.ops.Rot90Op
import java.io.ByteArrayOutputStream
import kotlin.math.max


class MainActivity : AppCompatActivity(), OnItemSelectedListener {
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
    private lateinit var context:Context;
    var isImport:Boolean=false
    val models= arrayOf("gen7.tflite", "gen8.tflite", "gen9.tflite", "yolo.tflite")
    fun loadModel(filename:String, context:Context){
        val model= FileUtil.loadMappedFile(context, filename)
        val interpreter= Interpreter(model)
        val inputShape = interpreter.getInputTensor(0).shape()
        this.inputImageWidth = inputShape[1]
        this.inputImageHeight = inputShape[2]
        modelInputSize = FLOAT_TYPE_SIZE * inputImageWidth *
                inputImageHeight * PIXEL_SIZE
        this.interpreter=interpreter;
    }
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding= ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        context=this
        loadModel("gen7.tflite", context)

        val spinner=binding.spinner
        spinner.onItemSelectedListener=this
        val ad: ArrayAdapter<*> = ArrayAdapter<Any?>(this, android.R.layout.simple_spinner_item, models)
        ad.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        spinner.adapter = ad

        cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        if (allPermissionsGranted()) {
                cameraProviderFuture.addListener(Runnable {
                    val cameraProvider = cameraProviderFuture.get()
                    bindPreview(cameraProvider)
                }, ContextCompat.getMainExecutor(this))
        } else {
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS);
        }


        binding.button.setOnClickListener{
            if(!isImport) {
                isImport=true
                val intent = Intent(MediaStore.ACTION_PICK_IMAGES)
                startActivityForResult(intent, 1)
                binding.button.text="Back to Preview"
            }else{
                isImport=false
                binding.button.text="Import image"
            }
        }
    }
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (resultCode === RESULT_OK) {
            // compare the resultCode with the
            // constant
            if (requestCode === 1) {
                // Get the url of the image from data
                val selectedImageUri: Uri = data?.data!!
                if (null != selectedImageUri) {
                    // update the image view in the layout
                    //binding.imageView.setImageURI(selectedImageUri)
                    val bitmap=MediaStore.Images.Media.getBitmap(this.getContentResolver(),selectedImageUri)
                    binding.textView.text=classify(bitmap)

                }
            }
        }
    }
    override fun onItemSelected(parent: AdapterView<*>?, view: View,
                                position: Int,id: Long)
    {
        loadModel(models[position],context)
        isImport=false
        binding.button.text="Import image"
    }

    override fun onNothingSelected(parent: AdapterView<*>?) {}
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
            if (!isImport) {
                val bitmap = imageProxyToBitmap(imageProxy)
                if (bitmap != null) {
                    val label = classify(bitmap)
                    runOnUiThread {
                        binding.textView.text = label
                    }
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
        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
    }
    override fun onDestroy(){
        super.onDestroy()
        this.interpreter.close()
    }

    private fun classify(bitmap: Bitmap): String {

        // TODO: Add code to run inference with TF Lite.
        // Pre-processing: resize the input image to match the model input shape.
        val smallest= Math.min(bitmap.width,bitmap.height)
        var ip:ImageProcessor.Builder = ImageProcessor.Builder()
        if(!isImport){ip=ip.add(Rot90Op(3))}
        ip=ip.add(ResizeWithCropOrPadOp(smallest, smallest))
        ip=ip.add(ResizeOp(inputImageWidth, inputImageHeight, ResizeOp.ResizeMethod.BILINEAR))
        val imageProcessor=ip.build()

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
            "Image Size: %d x %d\nPrediction Result: %s\nBroccoli: %2f\nCauliflower: %2f\nUnknown: %2f"
                .format(inputImageWidth,inputImageHeight,classes[maxIndex], result[0],result[1],result[2])

        return resultString
    }


}