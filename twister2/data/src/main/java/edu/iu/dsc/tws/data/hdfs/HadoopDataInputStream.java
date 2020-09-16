//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
package edu.iu.dsc.tws.data.hdfs;

import java.io.IOException;
import java.io.Reader;
import java.util.logging.Logger;

import javax.annotation.Nonnull;

import edu.iu.dsc.tws.api.data.FSDataInputStream;

public final class HadoopDataInputStream extends FSDataInputStream {

  private static final Logger LOG = Logger.getLogger(HadoopDataInputStream.class.getName());

  public static final int MIN_SKIP_BYTES = 1024 * 1024;

  private final org.apache.hadoop.fs.FSDataInputStream fosInputStream;

  public HadoopDataInputStream(org.apache.hadoop.fs.FSDataInputStream dataInputStream) {
    this.fosInputStream = dataInputStream;
  }

  @Override
  public void seek(long seekPosition) throws IOException {
    fosInputStream.seek(seekPosition);
  }

  @Override
  public long getPos() throws IOException {
    return fosInputStream.getPos();
  }

  @Override
  public Reader getReader() {
    return null;
  }

  @Override
  public int read() throws IOException {
    return fosInputStream.read();
  }

  @Override
  public void close() throws IOException {
    fosInputStream.close();
  }

  @Override
  public int read(@Nonnull byte[] buffer, int offset, int length) throws IOException {
    return fosInputStream.read(buffer, offset, length);
  }

  @Override
  public int available() throws IOException {
    return fosInputStream.available();
  }

  @Override
  public long skip(long n) throws IOException {
    return fosInputStream.skip(n);
  }

  public org.apache.hadoop.fs.FSDataInputStream getHadoopInputStream() {
    return fosInputStream;
  }

  public void forceSeek(long seekPos) throws IOException {
    fosInputStream.seek(seekPos);
  }

  public void skipFully(long bytes) throws IOException {
    /*while (bytes > 0) {
      bytes -= fosInputStream.skip(bytes);
    }*/
  }
}
