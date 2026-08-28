package puretf.chart;

import javax.imageio.ImageIO;
import javax.swing.JFrame;
import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Font;
import java.awt.FontMetrics;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.awt.geom.RoundRectangle2D;
import java.io.File;
import java.io.IOException;

/**
 * Evaluation view for image classifiers: a grid of sample images (one per
 * class) with the true label, predicted label and the class accuracy
 * percentage under each image. Images are laid out 5 per row. Writes a
 * composite PNG; optionally opens a Swing window.
 */
public final class EvalCharts {

    /** Image scale; 28px images render at SCALE px per pixel. */
    private static final int SCALE = 6;
    /** Images per row. */
    private static final int COLS = 5;
    private static final int MARGIN = 14;
    /** Vertical space reserved for the title above the first row. */
    private static final int TOPSPACE = 36;
    /** Total width of one grid cell (image + side padding). */
    private static final int CELL_W = 200;
    /** Line height used for the caption lines. */
    private static final int LINE_H = 24;
    /** Vertical space reserved under each image for the caption lines. */
    private static final int CAPTION_H = 80;
    /** Corner radius used for the rounded image tiles. */
    private static final int CORNER = 16;
    private static final Font LABEL_FONT =
            new Font(Font.SANS_SERIF, Font.BOLD, 16);
    private static final Font TITLE_FONT =
            new Font(Font.SANS_SERIF, Font.BOLD, 18);

    private EvalCharts() {}

    /**
     * @param images    flat test set, row-major [n][side*side]
     * @param imgSide   image width/height (e.g. 28)
     * @param sampleIdx index into the test set for EACH class
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
        int rows = (classes + COLS - 1) / COLS;
        int cell = imgSide * SCALE;
        int W = COLS * CELL_W + MARGIN * 2;
        int H = TOPSPACE + rows * (cell + CAPTION_H) + MARGIN;

        BufferedImage canvas = new BufferedImage(W, H,
                BufferedImage.TYPE_INT_RGB);
        Graphics2D g = canvas.createGraphics();
        try {
            g.setColor(Color.WHITE);
            g.fillRect(0, 0, W, H);
            g.setFont(TITLE_FONT);
            g.setColor(new Color(60, 60, 60));
            g.drawString(title, MARGIN, 26);
            paintGrid(g, MARGIN, TOPSPACE, images, imgSide, sampleIdx, yTrue,
                    yPred, names, accPct);
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

    /** Draws one sample image per class + caption lines under it, 5 per row. */
    private static void paintGrid(Graphics2D g, int x0, int y0,
                                  float[][] images, int side, int[] sampleIdx,
                                  int[] yTrue, int[] yPred, String[] names,
                                  double[] accPct) {
        int classes = names.length;
        int cell = side * SCALE;
        g.setFont(LABEL_FONT);
        FontMetrics fm = g.getFontMetrics();
        for (int idx = 0; idx < classes; idx++) {
            int r = idx / COLS, c = idx % COLS;
            int x = x0 + c * CELL_W + (CELL_W - cell) / 2;
            int y = y0 + r * (cell + CAPTION_H);
            BufferedImage img = toImage(images[sampleIdx[idx]], side);
            g.setColor(Color.BLACK);
            g.drawImage(img, x, y, cell, cell, null);
            g.setColor(Color.WHITE);
            g.fillRect(x, y, cell, CORNER / 2);
            g.fillRect(x, y + cell - CORNER / 2 - 1, cell, CORNER / 2 + 1);
            g.fillRect(x, y, CORNER / 2, cell);
            g.fillRect(x + cell - CORNER / 2 - 1, y, CORNER / 2 + 1, cell);

            RoundRectangle2D border = new RoundRectangle2D.Float(
                    x, y, cell, cell, CORNER, CORNER);
            g.setColor(Color.BLACK);
            g.draw(border);

            String trueL = shortName(names[yTrue[sampleIdx[idx]]]);
            String predL = "pred " + shortName(names[yPred[sampleIdx[idx]]]);
            String accL = String.format("%.1f%%", accPct[idx]);

            int cx = x0 + c * CELL_W;
            int base = y + cell;
            // Centre the caption text within the cell.
            int tx = cx + (CELL_W - fm.stringWidth(trueL)) / 2;
            g.setColor(Color.BLACK);
            g.drawString(trueL, tx, base + LINE_H);
            tx = cx + (CELL_W - fm.stringWidth(predL)) / 2;
            g.setColor(new Color(160, 30, 30));
            g.drawString(predL, tx, base + 2 * LINE_H);
            tx = cx + (CELL_W - fm.stringWidth(accL)) / 2;
            g.setColor(new Color(20, 90, 40));
            g.drawString(accL, tx, base + 3 * LINE_H);
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
