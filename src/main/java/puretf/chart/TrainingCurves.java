package puretf.chart;

import puretf.nn.History;

import javax.imageio.ImageIO;
import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

/**
 * Dependency-free training-curve plotter: renders loss/accuracy history to
 * a PNG (pure Java2D, no JFreeChart). Used after fitHistory().
 */
public final class TrainingCurves {

    /** Writes loss[] and accuracy[] (optionally val_*) as a 1000x600 PNG. */
    public static void save(String title, History h, String path)
            throws IOException {
        double[] loss = h.loss();
        double[] acc = h.acc();
        double[] vl = h.valLoss();
        double[] va = h.valAcc();

        int W = 1000, H = 600, M = 60;
        BufferedImage img = new BufferedImage(W, H, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = img.createGraphics();
        g.setColor(Color.WHITE);
        g.fillRect(0, 0, W, H);
        g.setColor(Color.BLACK);
        g.drawString(title, M, 28);

        int n = loss.length;
        double maxL = 1e-9;
        for (double v : loss) maxL = Math.max(maxL, v);
        if (vl != null) for (double v : vl) maxL = Math.max(maxL, v);
        int plotW = W - 2 * M, plotH = H - 2 * M;

        // grid + axes
        g.setColor(new Color(0xE0E0E0));
        for (int i = 0; i <= 5; i++) {
            int y = H - M - plotH * i / 5;
            g.drawLine(M, y, W - M, y);
            g.drawString(String.format("%.2f", maxL * i / 5), 12, y + 4);
            double av = 1.0 - i / 5.0;
            g.drawString(String.format("%.2f", av), W - M + 8, y + 4);
        }
        g.setColor(Color.BLACK);
        g.drawLine(M, H - M, W - M, H - M);
        g.drawLine(M, M, M, H - M);

        drawSeries(g, loss, maxL, M, plotW, plotH, H, n, Color.RED);
        if (vl != null) drawSeries(g, vl, maxL, M, plotW, plotH, H, n,
                new Color(0xFF8080));
        drawSeries(g, acc, 1.0, M, plotW, plotH, H, n, Color.BLUE);
        if (va != null) drawSeries(g, va, 1.0, M, plotW, plotH, H, n,
                new Color(0x8080FF));

        // legend
        g.setColor(Color.RED);
        g.drawString("loss", M + 4, M + 14);
        g.setColor(new Color(0xFF8080));
        g.drawString("val_loss", M + 44, M + 14);
        g.setColor(Color.BLUE);
        g.drawString("accuracy", M + 104, M + 14);
        g.setColor(new Color(0x8080FF));
        g.drawString("val_acc", M + 164, M + 14);

        g.dispose();
        File f = new File(path);
        f.getParentFile().mkdirs();
        ImageIO.write(img, "png", f);
    }

    private static void drawSeries(Graphics2D g, double[] v, double scale,
                                   int m, int pw, int ph, int h, int n,
                                   Color c) {
        if (v == null || v.length < 2) return;
        g.setStroke(new BasicStroke(2f));
        g.setColor(c);
        for (int i = 1; i < v.length; i++) {
            int x0 = m + (i - 1) * pw / (n - 1);
            int x1 = m + i * pw / (n - 1);
            g.drawLine(x0, h - m - (int) (Math.min(1.0, v[i - 1] / scale)
                            * ph),
                    x1, h - m - (int) (Math.min(1.0, v[i] / scale) * ph));
        }
    }

    private TrainingCurves() {}
}
