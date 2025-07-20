package com.example.tensorflowdemo

import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.Matrix
import android.media.ExifInterface
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import com.example.tensorflowdemo.databinding.ActivityMainBinding
import com.example.tensorflowdemo.ml.Model
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.nio.ByteBuffer
import java.nio.ByteOrder


class MainActivity : AppCompatActivity() {
    lateinit var interpreter:Interpreter;
    var inputImageWidth:Int= 0;
    var inputImageHeight:Int=0;
    var modelInputSize: Int = 0 // will be inferred from TF Lite model.
    private val FLOAT_TYPE_SIZE = 4
    private val PIXEL_SIZE = 1
    lateinit var binding:ActivityMainBinding;
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

        var btn=binding.button
        btn.setOnClickListener{
            val intent= Intent(MediaStore.ACTION_PICK_IMAGES)
            startActivityForResult(intent,1)
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
    override fun onDestroy(){
        super.onDestroy()
        this.interpreter.close()
    }

    private fun classify(bitmap: Bitmap): String {

        val model = Model.newInstance(applicationContext)
        // Creates inputs for reference.
        val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 100, 100, 3), DataType.FLOAT32)
        inputFeature0.loadBuffer(bitmapToNormalizedByteBuffer(cropCenterSquare(bitmap)))
        binding.imageView.setImageBitmap(byteBufferToBitmap(bitmapToNormalizedByteBuffer(cropCenterSquare(bitmap))))

        // Runs model inference and gets result.
        val outputs = model.process(inputFeature0)
        val result = outputs.outputFeature0AsTensorBuffer.floatArray

        // Releases model resources if no longer used.
        model.close()
        val maxIndex = result.indices.maxByOrNull { result[it] } ?: -1
        val classes=arrayOf("Broccoli","Cauliflower","Unknown")
        val resultString =
            "Prediction Result: %s\nConfidence: %2f"
                .format(classes[maxIndex], result[maxIndex])

        return resultString
    }

    fun cropCenterSquare(bitmap: Bitmap): Bitmap {
        val width = bitmap.width
        val height = bitmap.height
        val newEdge = minOf(width, height)

        val xOffset = (width - newEdge) / 2
        val yOffset = (height - newEdge) / 2

        return Bitmap.createBitmap(bitmap, xOffset, yOffset, newEdge, newEdge)
    }

    fun byteBufferToBitmap(buffer: ByteBuffer, width: Int = 100, height: Int = 100): Bitmap {
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        buffer.rewind()

        for (y in 0 until height) {
            for (x in 0 until width) {
                // Read normalized RGB floats
                val r = (buffer.float * 255.0f).toInt().coerceIn(0, 255)
                val g = (buffer.float * 255.0f).toInt().coerceIn(0, 255)
                val b = (buffer.float * 255.0f).toInt().coerceIn(0, 255)

                // Reconstruct ARGB pixel (opaque)
                val color = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
                bitmap.setPixel(x, y, color)
            }
        }

        return bitmap
    }

    fun bitmapToNormalizedByteBuffer(bitmap: Bitmap): ByteBuffer {
        val width = 100
        val height = 100
        val bytesPerChannel = 4  // Float = 4 bytes
        val numChannels = 3      // RGB
        val byteBuffer = ByteBuffer.allocateDirect(width * height * numChannels * bytesPerChannel)
        byteBuffer.order(ByteOrder.nativeOrder())

        // Resize the bitmap
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, width, height, true)

        // Extract pixels
        val pixels = IntArray(width * height)
        resizedBitmap.getPixels(pixels, 0, width, 0, 0, width, height)

        // Normalize and write to ByteBuffer
        for (pixel in pixels) {
            val r = ((pixel shr 16) and 0xFF) / 255.0f
            val g = ((pixel shr 8) and 0xFF) / 255.0f
            val b = (pixel and 0xFF) / 255.0f

            byteBuffer.putFloat(r)
            byteBuffer.putFloat(g)
            byteBuffer.putFloat(b)
        }

        byteBuffer.rewind()
        return byteBuffer
    }
}