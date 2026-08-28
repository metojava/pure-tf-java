package puretf.data;

import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.zip.GZIPInputStream;

/**
 * Pure-Java reader for the MNIST/Fashion-MNIST IDX file format (gzipped).
 * Images are returned normalized to [0,1] floats, row-major per image.
 */
public final class IdxDataset {

    private IdxDataset() {}

    public record Images(int count, int rows, int cols, float[][] pixels) {}

    public record Labels(int count, int[] labels) {}

    public static Images loadImages(String path) throws IOException {
        try (DataInputStream in = open(path)) {
            int magic = in.readInt();
            if (magic != 2051) throw new IOException("bad image magic " + magic);
            int count = in.readInt();
            int rows = in.readInt();
            int cols = in.readInt();
            byte[] raw = in.readNBytes(count * rows * cols);
            if (raw.length != count * rows * cols)
                throw new IOException("truncated image data");
            float[][] px = new float[count][rows * cols];
            for (int i = 0; i < count; i++)
                for (int p = 0; p < rows * cols; p++)
                    px[i][p] = (raw[i * rows * cols + p] & 0xFF) / 255f;
            return new Images(count, rows, cols, px);
        }
    }

    public static Labels loadLabels(String path) throws IOException {
        try (DataInputStream in = open(path)) {
            int magic = in.readInt();
            if (magic != 2049) throw new IOException("bad label magic " + magic);
            int count = in.readInt();
            byte[] raw = in.readNBytes(count);
            if (raw.length != count) throw new IOException("truncated labels");
            int[] labels = new int[count];
            for (int i = 0; i < count; i++) labels[i] = raw[i] & 0xFF;
            return new Labels(count, labels);
        }
    }

    private static DataInputStream open(String path) throws IOException {
        InputStream in = Files.newInputStream(Path.of(path));
        if (path.endsWith(".gz")) in = new GZIPInputStream(in);
        return new DataInputStream(new BufferedInputStream(in));
    }
}
