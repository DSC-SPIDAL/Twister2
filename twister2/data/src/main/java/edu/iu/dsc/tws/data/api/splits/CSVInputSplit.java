package edu.iu.dsc.tws.data.api.splits;

import java.io.IOException;
import java.nio.charset.Charset;
import java.util.Arrays;
import java.util.logging.Logger;

import edu.iu.dsc.tws.api.config.Config;
import edu.iu.dsc.tws.api.data.Path;
import edu.iu.dsc.tws.data.api.formatters.FileInputPartitioner;

/**
 * This class is primarily responsible for reading the split of a CSV file. It receive the input as
 * start and length of the file to be read for each split.
 */
public class CSVInputSplit extends FileInputSplit<Object> {

  private static final Logger LOG = Logger.getLogger(CSVInputSplit.class.getName());

  // The charset used to convert strings to bytes
  private String charsetName = "UTF-8";

  public static final String DEFAULT_LINE_DELIMITER = "\n";

  public static final String DEFAULT_FIELD_DELIMITER = ",";

  public static final String DEFAULT_TAB_DELIMITER = "\t";

  private static final byte CARRIAGE_RETURN = (byte) '\r';

  private static final byte NEW_LINE = (byte) '\n';

  private transient Charset charset;

  /**
   * The default read buffer size = 1MB.
   */
  private static final int DEFAULT_READ_BUFFER_SIZE = 1024 * 1024;

  /**
   * The configuration key to set the record delimiter.
   */
  protected static final String RECORD_DELIMITER = "delimited-format.delimiter";

  private byte[] delimiter = new byte[]{'\n'};

  private String delimiterString = null;

  private int lineLengthLimit = Integer.MAX_VALUE;

  private int bufferSize = -1;

  private transient byte[] readBuffer;
  private transient byte[] wrapBuffer;
  private transient byte[] currBuffer;

  private transient int readPos;
  private transient int limit;
  private transient int currOffset;      // offset in above buffer
  private transient int currLen;        // length of current byte sequence

  private transient boolean overLimit;
  private transient boolean end;

  private long offset = -1;

  private Config config;

  /**
   * Constructs a split with host information.
   *
   * @param num the number of this input split
   * @param file the file name
   * @param start the position of the first byte in the file to process
   * @param length the number of bytes in the file to process (-1 is flag for "read whole file")
   * @param hosts the list of hosts containing the block, possibly <code>null</code>
   */
  public CSVInputSplit(int num, Path file, long start, long length, String[] hosts) {
    super(num, file, start, length, hosts);
  }

  public byte[] getDelimiter() {
    return delimiter;
  }

  public void setDelimiter(byte[] delimiter) {
    if (delimiter == null) {
      throw new IllegalArgumentException("Delimiter must not be null");
    }
    this.delimiter = delimiter;
  }

  public String getDelimiterString() {
    return delimiterString;
  }

  public void setDelimiterString(String delimiterString) {
    if (delimiter == null) {
      throw new IllegalArgumentException("Delimiter must not be null");
    }
    this.delimiter = delimiterString.getBytes(getCharset());
    this.delimiterString = delimiterString;
  }

  public int getBufferSize() {
    return bufferSize;
  }

  public void setBufferSize(int bufferSize) {
    if (bufferSize < 2) {
      throw new IllegalArgumentException("Buffer size must be at least 2.");
    }
    this.bufferSize = bufferSize;
  }

  public String getCharsetName() {
    return charsetName;
  }

  public void setCharsetName(String charsetName) {
    this.charsetName = charsetName;
  }

  public int getLineLengthLimit() {
    return lineLengthLimit;
  }

  public void setLineLengthLimit(int lineLengthLimit) {
    this.lineLengthLimit = lineLengthLimit;
  }

  /**
   * Get the character set used for the row delimiter.
   *
   * @return the charset
   */
  public Charset getCharset() {
    if (this.charset == null) {
      this.charset = Charset.forName(charsetName);
    }
    return this.charset;
  }

  /**
   * Configures this input format by reading the path to the file from the
   * configuration and the string that
   * defines the record delimiter.
   *
   * @param parameters The configuration object to read the parameters from.
   */
  @Override
  public void configure(Config parameters) {
    super.configure(parameters);
    this.config = parameters;
    if (Arrays.equals(delimiter, new byte[]{'\n'})) {
      String delimString = parameters.getStringValue(RECORD_DELIMITER, null);
      if (delimString != null) {
        setDelimiterString(delimString);
      }
    }
  }

  /**
   * Opens the given input split. This method opens the input stream to the specified file,
   * allocates read buffers
   * and positions the stream at the correct position, making sure that any partial
   * record at the beginning is skipped.
   */
  public void open(Config cfg) throws IOException {
    super.open(cfg);
    initBuffers();
    this.offset = splitStart;
    if (this.splitStart != 0) {
      this.stream.seek(offset);
      if (this.overLimit) {
        this.end = true;
      }
    } else {
      fillBuffer(0);
    }
  }

  @Override
  public String nextRecord(Object record) throws IOException {
    if (readLine()) {
      return readRecord(this.currBuffer, this.currOffset, this.currLen);
    } else {
      this.end = true;
      return null;
    }
  }

  public String readRecord(byte[] bytes, int readOffset, int numBytes)
      throws IOException {
    int curNumBytes = numBytes;
    if (this.getDelimiter() != null && this.getDelimiter().length == 1
        && this.getDelimiter()[0] == NEW_LINE && readOffset + curNumBytes >= 1
        && bytes[readOffset + curNumBytes - 1] == CARRIAGE_RETURN) {
      curNumBytes -= 1;
    }
    return new String(bytes, readOffset, curNumBytes, this.charsetName);
  }

  protected final boolean readLine() throws IOException {
    if (this.stream == null || this.overLimit) {
      return false;
    }

    int countInWrapBuffer = 0;
    int delimPos = 0;
    while (true) {
      if (this.readPos >= this.limit) {
        if (!fillBuffer(delimPos)) {
          int countInReadBuffer = delimPos;
          if (countInWrapBuffer + countInReadBuffer > 0) {
            if (countInReadBuffer > 0) {
              if (this.wrapBuffer.length - countInWrapBuffer < countInReadBuffer) {
                byte[] tmp = new byte[countInWrapBuffer + countInReadBuffer];
                System.arraycopy(this.wrapBuffer, 0, tmp, 0, countInWrapBuffer);
                this.wrapBuffer = tmp;
              }
              System.arraycopy(this.readBuffer, 0, this.wrapBuffer,
                  countInWrapBuffer, countInReadBuffer);
              countInWrapBuffer += countInReadBuffer;
            }

            this.offset += countInWrapBuffer;
            setResult(this.wrapBuffer, 0, countInWrapBuffer);
            return true;
          } else {
            return true;
          }
        }
      }

      int startPos = this.readPos - delimPos;
      int count;
      // Search for next occurence of delimiter in read buffer.
      while (this.readPos < this.limit && delimPos < this.delimiter.length) {
        if ((this.readBuffer[this.readPos]) == this.delimiter[delimPos]) {
          delimPos++;
        } else {
          readPos -= delimPos;
          delimPos = 0;
        }
        readPos++;
      }
      if (readPos == limit && this.splitLength == 0) {
        this.end = true;
      }
      if (delimPos == this.delimiter.length) {
        int readBufferBytesRead = this.readPos - startPos;
        this.offset += countInWrapBuffer + readBufferBytesRead;
        count = readBufferBytesRead - this.delimiter.length;

        if (countInWrapBuffer > 0) {
          if (this.wrapBuffer.length < countInWrapBuffer + count) {
            final byte[] nb = new byte[countInWrapBuffer + count];
            System.arraycopy(this.wrapBuffer, 0, nb, 0, countInWrapBuffer);
            this.wrapBuffer = nb;
          }
          if (count >= 0) {
            System.arraycopy(this.readBuffer, 0, this.wrapBuffer, countInWrapBuffer, count);
          }
          setResult(this.wrapBuffer, 0, countInWrapBuffer + count);
          return true;
        } else {
          setResult(this.readBuffer, startPos, count);
          return true;
        }
      } else {
        count = this.limit - startPos;
        if (((long) countInWrapBuffer) + count > this.lineLengthLimit) {
          throw new IOException("The record length exceeded the maximum record length ("
              + this.lineLengthLimit + ").");
        }
        int bytesToMove = count - delimPos;
        if (this.wrapBuffer.length - countInWrapBuffer < bytesToMove) {
          byte[] tmp = new byte[Math.max(this.wrapBuffer.length * 2,
              countInWrapBuffer + bytesToMove)];
          System.arraycopy(this.wrapBuffer, 0, tmp, 0, countInWrapBuffer);
          this.wrapBuffer = tmp;
        }

        System.arraycopy(this.readBuffer, startPos, this.wrapBuffer,
            countInWrapBuffer, bytesToMove);
        countInWrapBuffer += bytesToMove;
        System.arraycopy(this.readBuffer, this.readPos - delimPos, this.readBuffer, 0, delimPos);
      }
    }
  }

  private void initBuffers() {
    this.bufferSize = this.bufferSize <= 0 ? DEFAULT_READ_BUFFER_SIZE : this.bufferSize;

    if (this.bufferSize <= this.delimiter.length) {
      throw new IllegalArgumentException("Buffer size must be greater than length of delimiter.");
    }

    if (this.readBuffer == null || this.readBuffer.length != this.bufferSize) {
      this.readBuffer = new byte[this.bufferSize];
    }
    if (this.wrapBuffer == null || this.wrapBuffer.length < 256) {
      this.wrapBuffer = new byte[256];
    }
    this.readPos = 0;
    this.limit = 0;
    this.overLimit = false;
    this.end = false;
  }

  /**
   * Fills the read buffer with bytes read from the file starting from an offset.
   */
  private boolean fillBuffer(int fillOffset) throws IOException {
    int maxReadLength = this.readBuffer.length - fillOffset;
    if (this.splitLength == FileInputPartitioner.READ_WHOLE_SPLIT_FLAG) {
      int read = this.stream.read(this.readBuffer, fillOffset, maxReadLength);
      if (read == -1) {
        this.stream.close();
        this.stream = null;
        return false;
      } else {
        this.readPos = fillOffset;
        this.limit = read;
        return true;
      }
    }

    int toRead;
    if (this.splitLength > 0) {
      toRead = this.splitLength > maxReadLength ? maxReadLength : (int) this.splitLength;
    } else {
      toRead = maxReadLength;
      this.overLimit = true;
    }

    int read = this.stream.read(this.readBuffer, fillOffset, toRead);
    if (read == -1) {
      this.stream.close();
      this.stream = null;
      return false;
    } else {
      this.splitLength -= read;
      this.readPos = fillOffset; // position from where to start reading
      this.limit = read + fillOffset; // number of valid bytes in the read buffer
      return true;
    }
  }

  private void setResult(byte[] buffer, int resultOffset, int len) {
    this.currBuffer = buffer;
    this.currOffset = resultOffset;
    this.currLen = len;
  }

  /**
   * Closes the input by releasing all buffers and closing the file input stream.
   *
   * @throws IOException Thrown, if the closing of the file stream causes an I/O error.
   */
  @Override
  public void close() throws IOException {
    this.wrapBuffer = null;
    this.readBuffer = null;
    super.close();
  }

  /**
   * Checks whether the current split is at its end.
   *
   * @return True, if the split is at its end, false otherwise.
   */
  @Override
  public boolean reachedEnd() {
    return this.end;
  }
}

