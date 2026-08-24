package puretf.chart;

import javax.imageio.ImageIO;
import javax.swing.JFrame;
import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

/**
 * Evaluation view for image classifiers: a grid of sample images (one per
 * class) with the true label, predicted label and the class accuracy
 * percentage under each image. Writes a composite PNG; optionally opens a
 * Swing window.
 */
public final class EvalCharts {

    private static final int SCALE = 3;
    private static final int CELL_W = 104;
    private static final int CAPTION_H = 62;

    private EvalCharts() {}

    /**
     * @param images    flat test set, row-major [n][side*side]
     * @param imgSide   image width/height (e.g. 28)
     * @param sampleIdx index into the test set for EACH class (grid columns)
     * @param yTrue/yPred ground truth vs model prediction per test sample
     * @param names     class names
     * @param accPct    per-class accuracy in percent (0..100)
     * @param show      open a Swing window with the grid
     */
    public static void render(String title, float[][] images, int imgSide,
                              int[] sampleIdx, int[] yTrue, int[] yPred,
                              String[] names, double[] accPct,
                              boolean show, String outPath) throws IOException {
        int classes = names.length;
        int cell = imgSide * SCALE;
        int W = classes * CELL_W + 12;
        int H = cell + CAPTION_H + 26;

        BufferedImage canvas = new BufferedImage(W, H,
                BufferedImage.TYPE_INT_RGB);
        Graphics2D g = canvas.createGraphics();
        try {
            g.setColor(Color.WHITE);
            g.fillRect(0, 0, W, H);
            paintGrid(g, 6, 20, images, imgSide, sampleIdx, yTrue, yPred,
                    names, accPct);
            g.setColor(new Color(60, 60, 60));
            g.drawString(title, 10, 14);
        } finally {
            g.dispose();
        }

        File out = new File(outPath);
        if (out.getParentFile() != null) out.getParentFile().mkdirs();
        ImageIO.write(canvas, "png", out);

        if (show && !java.awt.GraphicsEnvironment.isHeadless()) {
            JFrame f = new JFrame(title);
            f.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
            f.getContentPane().setLayout(new BorderLayout());
            f.add(new javax.swing.JLabel(new javax.swing.ImageIcon(canvas)),
                    BorderLayout.CENTER);
            f.pack();
            f.setVisible(true);
        }
    }

    /** Draws one sample image per class + caption lines under it. */
    private static void paintGrid(Graphics2D g, int x0, int y0,
                                  float[][] images, int side, int[] sampleIdx,
                                  int[] yTrue, int[] yPred, String[] names,
                                  double[] accPct) {
        int classes = names.length;
        int cell = side * SCALE;
        for (int c = 0; c < classes; c++) {
            int x = x0 + c * CELL_W + (CELL_W - cell) / 2;
            BufferedImage img = toImage(images[sampleIdx[c]], side);
            g.setColor(Color.BLACK);
            g.drawImage(img, x, y0, cell, cell, null);
            g.drawString(shortName(names[yTrue[sampleIdx[c]]]),
                    x, y0 + cell + 15);
            g.setColor(new Color(160, 30, 30));
            g.drawString("pred " + shortName(names[yPred[sampleIdx[c]]]),
                    x, y0 + cell + 29);
            g.setColor(new Color(20, 90, 40));
            g.drawString(String.format("%.1f%%", accPct[c]),
                    x, y0 + cell + 43);
            g.setColor(Color.BLACK);
        }
    }

    private static BufferedImage toImage(float[] pixels, int side) {
        BufferedImage img = new BufferedImage(side, side,
                BufferedImage.TYPE_BYTE_GRAY);
        for (int r = 0; r < side; r++)
            for (int c = 0; c < side; c++) {
                int v = (int) (pixels[r * side + c] * 255f);
                img.getRaster().setSample(c, r, 0, v);
            }
        return img;
    }

    private static String shortName(String s) {
        return s.length() <= 9 ? s : s.substring(0, 8) + ".";
    }
}
