package com.example.ihmdecodeur

import android.Manifest
import android.annotation.SuppressLint
import android.content.pm.ActivityInfo
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.SeekBar
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import com.example.ihmdecodeur.databinding.ActivityMainBinding
import org.opencv.android.OpenCVLoader
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.*

class MainActivity : AppCompatActivity(), SubtitleDecoder.Listener {

    private lateinit var binding: ActivityMainBinding
    private lateinit var cameraExecutor: ExecutorService
    private var subtitleDecoder: SubtitleDecoder? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        // Forcer le mode paysage
        requestedOrientation = ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE
        
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        if (!OpenCVLoader.initDebug()) {
            Log.e(TAG, "OpenCV initialization failed!")
            Toast.makeText(this, "Unable to load OpenCV", Toast.LENGTH_LONG).show()
        } else {
            Log.d(TAG, "OpenCV initialization successful!")
        }

        cameraExecutor = Executors.newSingleThreadExecutor()
        
        // Configuration du bouton paramètres
        binding.settingsButton.setOnClickListener {
            showSettingsDialog()
        }
        
        // Mettre à jour les statuts initiaux
        updateStatusDisplay()

        if (allPermissionsGranted()) {
            startCamera()
        } else {
            requestPermissions()
        }
    }
    
    private fun showSettingsDialog() {
        val dialogView = layoutInflater.inflate(R.layout.dialog_settings, null)
        val debugSwitch = dialogView.findViewById<androidx.appcompat.widget.SwitchCompat>(R.id.debug_switch)
        val pointSizeSeekBar = dialogView.findViewById<SeekBar>(R.id.point_size_seekbar)
        val pointSizeText = dialogView.findViewById<android.widget.TextView>(R.id.point_size_value)
        
        // Valeurs actuelles
        debugSwitch.isChecked = subtitleDecoder?.debugMode ?: false
        pointSizeSeekBar.progress = (subtitleDecoder?.pointSize ?: 6) - 1
        pointSizeText.text = "${subtitleDecoder?.pointSize ?: 6}px"
        
        pointSizeSeekBar.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                pointSizeText.text = "${progress + 1}px"
            }
            override fun onStartTrackingTouch(seekBar: SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: SeekBar?) {}
        })
        
        AlertDialog.Builder(this)
            .setTitle("⚙️ Paramètres")
            .setView(dialogView)
            .setPositiveButton("OK") { _, _ ->
                subtitleDecoder?.debugMode = debugSwitch.isChecked
                subtitleDecoder?.pointSize = pointSizeSeekBar.progress + 1
                updateStatusDisplay()
                Toast.makeText(this, "Paramètres mis à jour", Toast.LENGTH_SHORT).show()
            }
            .setNegativeButton("Annuler", null)
            .show()
    }
    
    private fun updateStatusDisplay() {
        val debugStatus = if (subtitleDecoder?.debugMode == true) "ON" else "OFF"
        val pointSize = subtitleDecoder?.pointSize ?: 6
        binding.statusText.text = "🔄 Redressement: ON | 🐛 Debug: $debugStatus | 🔴 Points: ${pointSize}px"
        
        // Cacher l'overlay debug si le mode debug est désactivé
        if (subtitleDecoder?.debugMode == false) {
            binding.debugOverlay.visibility = View.GONE
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder()
                .build()
                .also {
                    it.setSurfaceProvider(binding.cameraPreview.surfaceProvider)
                }

            val imageAnalyzer = ImageAnalysis.Builder()
                 .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also {
                    subtitleDecoder = SubtitleDecoder(this)
                    it.setAnalyzer(cameraExecutor, subtitleDecoder!!)
                }

            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    this, cameraSelector, preview, imageAnalyzer
                )
            } catch (exc: Exception) {
                Log.e(TAG, "Use case binding failed", exc)
            }

        }, ContextCompat.getMainExecutor(this))
    }

    private fun requestPermissions() {
        activityResultLauncher.launch(REQUIRED_PERMISSIONS)
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(
            baseContext, it
        ) == PackageManager.PERMISSION_GRANTED
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
    }

    private val activityResultLauncher =
        registerForActivityResult(
            ActivityResultContracts.RequestMultiplePermissions()
        ) { permissions ->
            if (permissions.getOrDefault(Manifest.permission.CAMERA, false)) {
                startCamera()
            } else {
                Toast.makeText(
                    baseContext,
                    "Camera permission is required",
                    Toast.LENGTH_SHORT
                ).show()
                finish()
            }
        }

    override fun onSubtitleDecoded(subtitle: String) {
        runOnUiThread {
            binding.subtitleText.text = subtitle
            updateStatusDisplay()
        }
    }
    
    override fun onDebugFrameAvailable(bitmap: android.graphics.Bitmap) {
        runOnUiThread {
            // Afficher la frame de debug par-dessus la preview
            binding.debugOverlay.setImageBitmap(bitmap)
            binding.debugOverlay.visibility = View.VISIBLE
        }
    }

    companion object {
        private const val TAG = "IHM-Decodeur"
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
    }
}

class SubtitleDecoder(private val listener: Listener) : ImageAnalysis.Analyzer {

    interface Listener {
        fun onSubtitleDecoded(subtitle: String)
        fun onDebugFrameAvailable(bitmap: android.graphics.Bitmap)
    }

    private val gridWidth = 16
    private val gridHeight = 16
    var pointSize = 6
    var debugMode = false

    // Positions des 4 grilles (identiques au Python)
    private val gridPositions = listOf(
        Point(0.002, 0.02),   // Haut gauche
        Point(0.52, 0.02),    // Haut droite
        Point(0.002, 0.52),   // Bas gauche
        Point(0.52, 0.52)     // Bas droite
    )
    
    // Offsets pour chaque grille
    private val gridOffsets = listOf(0, 5, 10, 15)

    private val detectionBuffer = ArrayDeque<String>(10)
    private var currentSubtitle = ""
    private var frameCount = 0
    
    // Variables pour le redressement de perspective
    private var perspectiveMatrix: Mat? = null
    private var screenCorners: MatOfPoint2f? = null
    private val cornerBuffer = ArrayDeque<MatOfPoint2f>(5)
    
    // Frame de debug pour affichage visuel
    private var debugFrame: Mat? = null
    private var lastDebugBitmap: android.graphics.Bitmap? = null

    @SuppressLint("UnsafeOptInUsageError")
    override fun analyze(imageProxy: ImageProxy) {
        frameCount++
        
        // Ne traiter qu'une frame sur 5 (comme dans le Python)
        if (frameCount % 5 != 0) {
            imageProxy.close()
            return
        }
        
        val mat = imageProxyToMat(imageProxy)
        // Pas de rotation - le téléphone est déjà en paysage
        
        // Appliquer le redressement de perspective (toujours actif)
        val correctedMat = applyPerspectiveCorrection(mat) ?: mat
        
        // Ajouter les contours rouges
        val borderedMat = addRedBorders(correctedMat)
        
        val positions = detectWhiteCircles(borderedMat)

        if (positions.isNotEmpty()) {
            val binaryStr = positionsToBinary(positions)
            val text = binaryToText(binaryStr)

            if (text.isNotBlank()) {
                if(detectionBuffer.size == 10) detectionBuffer.removeFirst()
                detectionBuffer.addLast(text)

                if (detectionBuffer.size >= 3) {
                    val mostCommon = detectionBuffer.groupingBy { it }.eachCount().maxByOrNull { it.value }?.key
                    if (mostCommon != null && mostCommon != currentSubtitle) {
                        currentSubtitle = mostCommon
                        listener.onSubtitleDecoded(currentSubtitle)
                    }
                }
            }
        }
        
        // 🎨 Envoyer la frame de debug à l'UI si le mode debug est actif
        if (debugMode && debugFrame != null) {
            val bitmap = matToBitmap(debugFrame!!)
            listener.onDebugFrameAvailable(bitmap)
        }
        
        if (correctedMat != mat) correctedMat.release()
        borderedMat.release()
        mat.release()
        imageProxy.close()
    }
    
    private fun addRedBorders(frame: Mat): Mat {
        val bordered = frame.clone()
        val height = frame.rows()
        val width = frame.cols()
        
        val thickness = 8
        val cornerSize = min(width, height) / 8
        val redColor = Scalar(0.0, 0.0, 255.0) // BGR
        
        // Coin haut-gauche
        Imgproc.rectangle(bordered, Point(0.0, 0.0), Point(cornerSize.toDouble(), thickness.toDouble()), redColor, -1)
        Imgproc.rectangle(bordered, Point(0.0, 0.0), Point(thickness.toDouble(), cornerSize.toDouble()), redColor, -1)
        
        // Coin haut-droite
        Imgproc.rectangle(bordered, Point((width - cornerSize).toDouble(), 0.0), Point(width.toDouble(), thickness.toDouble()), redColor, -1)
        Imgproc.rectangle(bordered, Point((width - thickness).toDouble(), 0.0), Point(width.toDouble(), cornerSize.toDouble()), redColor, -1)
        
        // Coin bas-gauche
        Imgproc.rectangle(bordered, Point(0.0, (height - thickness).toDouble()), Point(cornerSize.toDouble(), height.toDouble()), redColor, -1)
        Imgproc.rectangle(bordered, Point(0.0, (height - cornerSize).toDouble()), Point(thickness.toDouble(), height.toDouble()), redColor, -1)
        
        // Coin bas-droite
        Imgproc.rectangle(bordered, Point((width - cornerSize).toDouble(), (height - thickness).toDouble()), Point(width.toDouble(), height.toDouble()), redColor, -1)
        Imgproc.rectangle(bordered, Point((width - thickness).toDouble(), (height - cornerSize).toDouble()), Point(width.toDouble(), height.toDouble()), redColor, -1)
        
        return bordered
    }
    
    private fun applyPerspectiveCorrection(frame: Mat): Mat? {
        val corners = detectScreenCorners(frame)
        
        if (corners != null) {
            Log.d("SubtitleDecoder", "🎯 Coins d'écran détectés!")
            // Ajouter au buffer pour stabiliser
            if (cornerBuffer.size == 5) cornerBuffer.removeFirst()
            cornerBuffer.addLast(corners)
            
            if (cornerBuffer.size >= 3) {
                // Moyenne des détections
                val avgCorners = averageCorners(cornerBuffer.toList())
                
                // Garder les dimensions originales du frame pour éviter l'étirement
                val targetWidth = frame.cols()
                val targetHeight = frame.rows()
                
                val dstCorners = MatOfPoint2f(
                    Point(0.0, 0.0),
                    Point((targetWidth - 1).toDouble(), 0.0),
                    Point((targetWidth - 1).toDouble(), (targetHeight - 1).toDouble()),
                    Point(0.0, (targetHeight - 1).toDouble())
                )
                
                try {
                    perspectiveMatrix = Imgproc.getPerspectiveTransform(avgCorners, dstCorners)
                    val corrected = Mat()
                    Imgproc.warpPerspective(frame, corrected, perspectiveMatrix!!, 
                        org.opencv.core.Size(targetWidth.toDouble(), targetHeight.toDouble()),
                        Imgproc.INTER_LINEAR)
                    screenCorners = avgCorners
                    dstCorners.release()
                    
                    Log.d("SubtitleDecoder", "✅ Redressement appliqué: ${frame.cols()}x${frame.rows()} (dimensions conservées)")
                    
                    return corrected
                } catch (e: Exception) {
                    Log.e("SubtitleDecoder", "❌ Perspective transform error: ${e.message}")
                }
                dstCorners.release()
            }
        } else if (perspectiveMatrix != null) {
            // Utiliser la dernière matrice stable
            try {
                val corrected = Mat()
                Imgproc.warpPerspective(frame, corrected, perspectiveMatrix!!, frame.size(), Imgproc.INTER_LINEAR)
                return corrected
            } catch (e: Exception) {
                perspectiveMatrix = null
            }
        }
        
        return null
    }
    
    private fun detectScreenCorners(frame: Mat): MatOfPoint2f? {
        val gray = Mat()
        Imgproc.cvtColor(frame, gray, Imgproc.COLOR_BGR2GRAY)
        
        // Pré-traitement
        Imgproc.medianBlur(gray, gray, 5)
        val clahe = Imgproc.createCLAHE(2.0, org.opencv.core.Size(8.0, 8.0))
        clahe.apply(gray, gray)
        
        // Détection de contours multi-seuils
        val edges1 = Mat()
        val edges2 = Mat()
        val edges3 = Mat()
        Imgproc.Canny(gray, edges1, 15.0, 50.0)
        Imgproc.Canny(gray, edges2, 30.0, 100.0)
        Imgproc.Canny(gray, edges3, 50.0, 150.0)
        
        val edges = Mat()
        Core.bitwise_or(edges1, edges2, edges)
        Core.bitwise_or(edges, edges3, edges)
        
        // Morphologie
        val kernelDilate = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, org.opencv.core.Size(5.0, 5.0))
        val kernelClose = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, org.opencv.core.Size(9.0, 9.0))
        Imgproc.dilate(edges, edges, kernelDilate, Point(-1.0, -1.0), 2)
        Imgproc.morphologyEx(edges, edges, Imgproc.MORPH_CLOSE, kernelClose)
        
        // Trouver les contours
        val contours = mutableListOf<MatOfPoint>()
        val hierarchy = Mat()
        Imgproc.findContours(edges, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)
        
        val frameArea = (frame.rows() * frame.cols()).toDouble()
        val validContours = contours.filter { contour ->
            val area = Imgproc.contourArea(contour)
            area > 0.1 * frameArea && area < 0.9 * frameArea
        }.sortedByDescending { Imgproc.contourArea(it) }
        
        for (contour in validContours.take(3)) {
            val perimeter = Imgproc.arcLength(MatOfPoint2f(*contour.toArray()), true)
            for (epsilonFactor in listOf(0.015, 0.02, 0.025, 0.03, 0.04)) {
                val epsilon = epsilonFactor * perimeter
                val approx = MatOfPoint2f()
                Imgproc.approxPolyDP(MatOfPoint2f(*contour.toArray()), approx, epsilon, true)
                
                if (approx.rows() == 4) {
                    val ordered = orderCorners(approx)
                    if (isValidQuadrilateral(ordered)) {
                        // Libérer les ressources
                        gray.release()
                        edges1.release()
                        edges2.release()
                        edges3.release()
                        edges.release()
                        kernelDilate.release()
                        kernelClose.release()
                        hierarchy.release()
                        contours.forEach { it.release() }
                        approx.release()
                        return ordered
                    }
                    ordered.release()
                }
                approx.release()
            }
        }
        
        // Libérer les ressources
        gray.release()
        edges1.release()
        edges2.release()
        edges3.release()
        edges.release()
        kernelDilate.release()
        kernelClose.release()
        hierarchy.release()
        contours.forEach { it.release() }
        
        return null
    }
    
    private fun isValidQuadrilateral(corners: MatOfPoint2f): Boolean {
        val points = corners.toArray()
        if (points.size != 4) return false
        
        // Vérifier les longueurs des côtés
        val sides = (0..3).map { i ->
            norm(points[i], points[(i + 1) % 4])
        }
        
        val minSide = sides.minOrNull() ?: 0.0
        val maxSide = sides.maxOrNull() ?: 0.0
        
        if (minSide < 50.0 || maxSide / minSide > 2.5) return false
        
        // Vérifier les angles
        for (i in 0..3) {
            val p1 = points[(i - 1 + 4) % 4]
            val p2 = points[i]
            val p3 = points[(i + 1) % 4]
            
            val v1x = p1.x - p2.x
            val v1y = p1.y - p2.y
            val v2x = p3.x - p2.x
            val v2y = p3.y - p2.y
            
            val dot = v1x * v2x + v1y * v2y
            val norm1 = sqrt(v1x * v1x + v1y * v1y)
            val norm2 = sqrt(v2x * v2x + v2y * v2y)
            
            val cosAngle = (dot / (norm1 * norm2 + 1e-6)).coerceIn(-1.0, 1.0)
            val angle = Math.toDegrees(acos(cosAngle))
            
            if (angle < 60.0 || angle > 120.0) return false
        }
        
        return true
    }
    
    private fun orderCorners(corners: MatOfPoint2f): MatOfPoint2f {
        val points = corners.toArray()
        val sums = points.map { it.x + it.y }
        val diffs = points.map { it.y - it.x }
        
        val ordered = arrayOf(
            points[sums.indexOf(sums.minOrNull()!!)],  // top-left
            points[diffs.indexOf(diffs.minOrNull()!!)], // top-right
            points[sums.indexOf(sums.maxOrNull()!!)],  // bottom-right
            points[diffs.indexOf(diffs.maxOrNull()!!)], // bottom-left
        )
        
        return MatOfPoint2f(*ordered)
    }
    
    private fun averageCorners(cornersList: List<MatOfPoint2f>): MatOfPoint2f {
        val avgPoints = Array(4) { Point(0.0, 0.0) }
        
        for (corners in cornersList) {
            val points = corners.toArray()
            for (i in 0..3) {
                avgPoints[i].x += points[i].x
                avgPoints[i].y += points[i].y
            }
        }
        
        val count = cornersList.size.toDouble()
        for (i in 0..3) {
            avgPoints[i].x /= count
            avgPoints[i].y /= count
        }
        
        return MatOfPoint2f(*avgPoints)
    }
    
    private fun norm(p1: Point, p2: Point): Double {
        val dx = p2.x - p1.x
        val dy = p2.y - p1.y
        return sqrt(dx * dx + dy * dy)
    }

    private fun detectWhiteCircles(frame: Mat): List<Point> {
        // Créer une copie pour le debug visuel
        val workingFrame = if (debugMode) frame.clone() else frame
        
        val gray = Mat()
        Imgproc.cvtColor(frame, gray, Imgproc.COLOR_BGR2GRAY)
        
        // CLAHE pour améliorer le contraste
        val clahe = Imgproc.createCLAHE(3.0, org.opencv.core.Size(8.0, 8.0))
        clahe.apply(gray, gray)
        
        // Floutage gaussien
        val blurred = Mat()
        Imgproc.GaussianBlur(gray, blurred, org.opencv.core.Size(7.0, 7.0), 2.0)

        val allGridDetections = mutableListOf<List<Point>>()

        for ((gridId, gridPos) in gridPositions.withIndex()) {
            val relX = gridPos.x
            val relY = gridPos.y
            val gridPixelWidth = (frame.cols() * 0.48).toInt()
            val gridPixelHeight = (frame.rows() * 0.48).toInt()
            val gridXOffset = (relX * frame.cols()).toInt()
            val gridYOffset = (relY * frame.rows()).toInt()

            if (gridYOffset + gridPixelHeight > frame.rows() || gridXOffset + gridPixelWidth > frame.cols()) {
                allGridDetections.add(emptyList())
                continue
            }

            val gridRect = Rect(gridXOffset, gridYOffset, gridPixelWidth, gridPixelHeight)
            val gridBlurred = Mat(blurred, gridRect)

            // Paramètres adaptatifs selon la taille des points
            val (minRadius, maxRadius, minDist, param1, param2) = when (pointSize) {
                1 -> listOf(1, 3, 1.0, 10.0, 2.0)
                2 -> listOf(1, 4, 1.0, 15.0, 3.0)
                3 -> listOf(1, 6, 2.0, 20.0, 5.0)
                6 -> listOf(4, 8, 6.0, 37.0, 11.0)
                7 -> listOf(6, 8, 6.0, 45.0, 13.0)
                else -> listOf(max(1, pointSize - 2), pointSize + 3, max(5.0, (pointSize - 1).toDouble()), 37.0, 11.0)
            }

            // Détection de cercles
            val circles = Mat()
            Imgproc.HoughCircles(
                gridBlurred,
                circles,
                Imgproc.HOUGH_GRADIENT,
                1.0,
                minDist as Double,
                param1 as Double,
                param2 as Double,
                minRadius as Int,
                maxRadius as Int
            )
            
            // Fallback si très peu de cercles
            if (circles.cols() < 3) {
                Imgproc.HoughCircles(
                    gridBlurred,
                    circles,
                    Imgproc.HOUGH_GRADIENT,
                    1.0,
                    max(2.0, (minDist as Double) - 3.0),
                    max(25.0, (param1 as Double) - 10.0),
                    max(6.0, (param2 as Double) - 4.0),
                    max(1, (minRadius as Int) - 1),
                    (maxRadius as Int) + 2
                )
            }

            val gridDetections = mutableListOf<Point>()
            val cellWidth = gridPixelWidth.toDouble() / gridWidth
            val cellHeight = gridPixelHeight.toDouble() / gridHeight

            for (i in 0 until circles.cols()) {
                val circle = circles.get(0, i)
                val x = circle[0].toInt()
                val y = circle[1].toInt()
                val r = circle[2].toInt()

                // Filtrage des faux positifs
                if (r < pointSize * 0.63 || r > pointSize * 1.9) continue
                
                // Vérifier le contraste dans la région
                val y1 = max(0, y - r - 2)
                val y2 = min(gridBlurred.rows(), y + r + 3)
                val x1 = max(0, x - r - 2)
                val x2 = min(gridBlurred.cols(), x + r + 3)
                
                if (y2 - y1 < 3 || x2 - x1 < 3) continue
                
                val circleRegion = Mat(gridBlurred, Rect(x1, y1, x2 - x1, y2 - y1))
                val mean = MatOfDouble()
                val stdDev = MatOfDouble()
                Core.meanStdDev(circleRegion, mean, stdDev)
                val std = stdDev.get(0, 0)[0]
                
                mean.release()
                stdDev.release()
                
                if (std < 7.0) {
                    circleRegion.release()
                    continue
                }
                
                circleRegion.release()

                val gridX = (x / cellWidth).toInt()
                val gridY = (y / cellHeight).toInt()

                if (gridX in 0 until gridWidth && gridY in 0 until gridHeight) {
                    gridDetections.add(Point(gridX.toDouble(), gridY.toDouble()))
                    
                    // 🎨 DEBUG VISUEL : Dessiner les cercles détectés
                    if (debugMode) {
                        val absX = x + gridXOffset
                        val absY = y + gridYOffset
                        Imgproc.circle(workingFrame, Point(absX.toDouble(), absY.toDouble()), r, Scalar(0.0, 255.0, 0.0), 2)
                        Imgproc.putText(workingFrame, "G$gridId($gridX,$gridY)", 
                            Point(absX + 8.0, absY - 8.0),
                            Imgproc.FONT_HERSHEY_SIMPLEX, 0.3, Scalar(255.0, 255.0, 255.0), 1)
                    }
                }
            }
            
            allGridDetections.add(gridDetections)
            
            // 🎨 DEBUG VISUEL : Dessiner les grilles
            if (debugMode) {
                val gridColor = when(gridId) {
                    0 -> Scalar(255.0, 0.0, 0.0)      // Bleu
                    1 -> Scalar(0.0, 255.0, 0.0)      // Vert
                    2 -> Scalar(0.0, 0.0, 255.0)      // Rouge
                    else -> Scalar(255.0, 255.0, 0.0) // Cyan
                }
                Imgproc.rectangle(workingFrame, 
                    Point(gridXOffset.toDouble(), gridYOffset.toDouble()),
                    Point((gridXOffset + gridPixelWidth).toDouble(), (gridYOffset + gridPixelHeight).toDouble()),
                    gridColor, 2)
                    
                val params = when (pointSize) {
                    1 -> "1-3/1/10-2"
                    2 -> "1-4/1/15-3"
                    3 -> "1-6/2/20-5"
                    6 -> "4-8/6/37-11"
                    7 -> "6-8/6/45-13"
                    else -> "${max(1, pointSize - 2)}-${pointSize + 3}/${max(5, pointSize - 1)}/37-11"
                }
                Imgproc.putText(workingFrame, "Grid $gridId ($params)", 
                    Point(gridXOffset + 5.0, gridYOffset + 20.0),
                    Imgproc.FONT_HERSHEY_SIMPLEX, 0.3, gridColor, 1)
            }
            gridBlurred.release()
            circles.release()
        }

        gray.release()
        blurred.release()
        
        val votedPositions = voteMajority(allGridDetections)
        
        // 🎨 DEBUG : Stocker la frame pour affichage
        if (debugMode) {
            debugFrame?.release()
            debugFrame = workingFrame
            
            // Log des détections
            val totalDetections = allGridDetections.sumOf { it.size }
            Log.d("SubtitleDecoder", "📊 TOTAL: $totalDetections détections → ${votedPositions.size} validées")
            allGridDetections.forEachIndexed { i, detections ->
                Log.d("SubtitleDecoder", "   Grid $i: ${detections.size} détections")
            }
        } else {
            if (workingFrame != frame) workingFrame.release()
        }

        return votedPositions
    }

    private fun voteMajority(allGridDetections: List<List<Point>>): List<Point> {
        val positionVotes = mutableMapOf<Pair<Int, Int>, Int>()
        val positionsPerGrid = gridWidth * gridHeight
        
        for ((gridId, gridDetections) in allGridDetections.withIndex()) {
            val offset = gridOffsets[gridId]
            
            for (pos in gridDetections) {
                val x = pos.x.toInt()
                val y = pos.y.toInt()
                // Convertir en index linéaire
                val detectedIndex = y * gridWidth + x
                // Appliquer la dé-offset
                val originalIndex = (detectedIndex - offset + positionsPerGrid) % positionsPerGrid
                // Reconvertir en coordonnées
                val originalX = originalIndex % gridWidth
                val originalY = originalIndex / gridWidth
                val originalPos = Pair(originalX, originalY)
                
                positionVotes[originalPos] = (positionVotes[originalPos] ?: 0) + 1
            }
        }
        
        return positionVotes.filter { it.value >= 2 }
            .keys
            .map { Point(it.first.toDouble(), it.second.toDouble()) }
    }

    private fun positionsToBinary(positions: List<Point>): String {
        val binaryStr = CharArray(gridWidth * gridHeight) { '0' }
        for (pos in positions) {
            val index = pos.y.toInt() * gridWidth + pos.x.toInt()
            if (index < binaryStr.size) {
                binaryStr[index] = '1'
            }
        }
        return String(binaryStr)
    }

    private fun binaryToText(binaryStr: String): String {
        return binaryStr.chunked(8)
            .mapNotNull {
                if (it.length == 8) {
                    try {
                        val charCode = it.toInt(2)
                        if (charCode in 32..126) charCode.toChar() else null
                    } catch (e: NumberFormatException) {
                        null
                    }
                } else null
            }.joinToString("")
    }

    private fun imageProxyToMat(image: ImageProxy): Mat {
        val yBuffer = image.planes[0].buffer
        val uBuffer = image.planes[1].buffer
        val vBuffer = image.planes[2].buffer

        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()

        val nv21 = ByteArray(ySize + uSize + vSize)
        yBuffer.get(nv21, 0, ySize)
        vBuffer.get(nv21, ySize, vSize)
        uBuffer.get(nv21, ySize + vSize, uSize)

        val yuv = Mat(image.height + image.height / 2, image.width, CvType.CV_8UC1)
        yuv.put(0, 0, nv21)
        val mat = Mat()
        Imgproc.cvtColor(yuv, mat, Imgproc.COLOR_YUV2BGR_NV21, 3)
        yuv.release()
        return mat
    }
    
    private fun matToBitmap(mat: Mat): android.graphics.Bitmap {
        val bitmap = android.graphics.Bitmap.createBitmap(mat.cols(), mat.rows(), android.graphics.Bitmap.Config.ARGB_8888)
        org.opencv.android.Utils.matToBitmap(mat, bitmap)
        return bitmap
    }
}