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

import java.io.Closeable;
import java.io.IOException;
import java.net.URI;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Logger;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.LocatedFileStatus;
import org.apache.hadoop.fs.RemoteIterator;

import edu.iu.dsc.tws.api.data.BlockLocation;
import edu.iu.dsc.tws.api.data.FSDataOutputStream;
import edu.iu.dsc.tws.api.data.FileStatus;
import edu.iu.dsc.tws.api.data.FileSystem;
import edu.iu.dsc.tws.api.data.Path;
import static edu.iu.dsc.tws.data.utils.PreConditions.checkNotNull;

public class HadoopFileSystem extends FileSystem implements Closeable {

  private static final Logger LOG = Logger.getLogger(HadoopFileSystem.class.getName());

  private org.apache.hadoop.conf.Configuration conf;
  private org.apache.hadoop.fs.FileSystem hadoopFileSystem;

  public HadoopFileSystem(
      org.apache.hadoop.conf.Configuration hadoopConfig,
      org.apache.hadoop.fs.FileSystem hadoopfileSystem) {

    this.conf = checkNotNull(hadoopConfig, "hadoopConfig");
    this.hadoopFileSystem = checkNotNull(hadoopfileSystem, "fileSystem");
  }

  private static Class<? extends FileSystem> getFileSystemByName(String className)
      throws ClassNotFoundException {
    return Class.forName(className, true,
        FileSystem.class.getClassLoader()).asSubclass(FileSystem.class);
  }

  private static org.apache.hadoop.fs.Path toHadoopPath(Path path) {
    return new org.apache.hadoop.fs.Path(path.toUri());
  }

  public org.apache.hadoop.fs.FileSystem getHadoopFileSystem() {
    return this.hadoopFileSystem;
  }

  private Configuration getHadoopConfiguration() {
    return new Configuration();
  }

  /**
   * Get the working Directory
   */
  @Override
  public Path getWorkingDirectory() {
    return new Path(this.hadoopFileSystem.getWorkingDirectory().toUri());
  }

  /**
   * Set the working Directory
   */
  @Override
  public void setWorkingDirectory(Path path1) {
  }


  @Override
  public URI getUri() {
    return hadoopFileSystem.getUri();
  }

  /**
   * Called after a new FileSystem instance is constructed.
   *
   * @param name a {@link URI} whose authority section names the host, port, etc.
   * for this file system
   */
  @Override
  public void initialize(URI name) {
  }

  /**
   * It returns the status of the file respective to the path given by the user.
   *
   * @param f The path we want information from
   */
  @Override
  public FileStatus getFileStatus(Path f) throws IOException {
    org.apache.hadoop.fs.FileStatus status = this.hadoopFileSystem.getFileStatus(toHadoopPath(f));
    final FileStatus[] fileStatuses = listStatus(f);
    return fileStatuses[0];
  }

  @Override
  public BlockLocation[] getFileBlockLocations(final FileStatus file,
                                               final long start, final long len)
      throws IOException {
    if (!(file instanceof HadoopFileStatus)) {
      throw new IOException("file is not an instance of DistributedFileStatus");
    }

    final HadoopFileStatus f = (HadoopFileStatus) file;
    final org.apache.hadoop.fs.BlockLocation[] blkLocations =
        hadoopFileSystem.getFileBlockLocations(f.getInternalFileStatus(), start, len);
    final HadoopBlockLocation[] distBlkLocations = new HadoopBlockLocation[blkLocations.length];
    for (int i = 0; i < distBlkLocations.length; i++) {
      distBlkLocations[i] = new HadoopBlockLocation(blkLocations[i]);
    }
    return distBlkLocations;
  }

  @Override
  public HadoopDataInputStream open(final Path f, final int bufferSize) throws IOException {
    final org.apache.hadoop.fs.Path directoryPath = toHadoopPath(f);
    final org.apache.hadoop.fs.FSDataInputStream fsDataInputStream =
        this.hadoopFileSystem.open(directoryPath, bufferSize);
    return new HadoopDataInputStream(fsDataInputStream);
  }

  /**
   * This method open and return the input stream object respective to the path
   *
   * @param f Open an data input stream at the indicated path
   */
  @Override
  public HadoopDataInputStream open(final Path f) throws IOException {
    final org.apache.hadoop.fs.FSDataInputStream fsDataInputStream
        = hadoopFileSystem.open(toHadoopPath(f));
    return new HadoopDataInputStream(fsDataInputStream);
  }

  @Override
  public HadoopDataOutputStream create(final Path f) throws IOException {
    final org.apache.hadoop.fs.FSDataOutputStream fsDataOutputStream =
        this.hadoopFileSystem.create(toHadoopPath(f));
    return new HadoopDataOutputStream(fsDataOutputStream);
  }

  @Override
  public FSDataOutputStream create(Path f, WriteMode writeMode) throws IOException {
    final org.apache.hadoop.fs.FSDataOutputStream fsDataOutputStream =
        this.hadoopFileSystem.create(toHadoopPath(f), writeMode == WriteMode.OVERWRITE);
    return new HadoopDataOutputStream(fsDataOutputStream);
  }

  public HadoopDataOutputStream append(Path path) throws IOException {
    final org.apache.hadoop.fs.FSDataOutputStream fsDataOutputStream =
        this.hadoopFileSystem.append(toHadoopPath(path));
    return new HadoopDataOutputStream(fsDataOutputStream);
  }

  @Override
  public boolean delete(final Path f, final boolean recursive) throws IOException {
    return this.hadoopFileSystem.delete(toHadoopPath(f), recursive);
  }

  @Override
  public boolean exists(Path f) throws IOException {
    return this.hadoopFileSystem.exists(toHadoopPath(f));
  }

  @Override
  public FileStatus[] listStatus(final Path f) throws IOException {
    final org.apache.hadoop.fs.FileStatus[] hadoopFiles =
        this.hadoopFileSystem.listStatus(toHadoopPath(f));
    final FileStatus[] files = new FileStatus[hadoopFiles.length];
    for (int i = 0; i < files.length; i++) {
      files[i] = new HadoopFileStatus(hadoopFiles[i]);
    }
    return files;
  }

  @Override
  public boolean mkdirs(final Path f) throws IOException {
    return this.hadoopFileSystem.mkdirs(toHadoopPath(f));
  }

  @Override
  public boolean rename(final Path src, final Path dst) throws IOException {
    return this.hadoopFileSystem.rename(toHadoopPath(src), toHadoopPath(dst));
  }

  @SuppressWarnings("deprecation")
  @Override
  public long getDefaultBlockSize() {
    return this.hadoopFileSystem.getDefaultBlockSize();
  }

  @Override
  public boolean isDistributedFS() {
    return true;
  }

  /**
   * List the statuses of the files/directories in the given path if the path is
   * a directory.
   *
   * @param f given path
   * @return the statuses of the files/directories in the given patch
   */
  @Override
  public FileStatus[] listFiles(Path f) throws IOException {
    RemoteIterator<LocatedFileStatus> listFiles = this.hadoopFileSystem.listFiles(
        toHadoopPath(f), true);
    List<FileStatus> statusList = new ArrayList<>();
    while (listFiles.hasNext()) {
      LocatedFileStatus next = listFiles.next();
      FileStatus status = new HadoopFileStatus(next);
      statusList.add(status);
    }
    return statusList.toArray(new FileStatus[0]);
  }

  public void close() throws IOException {
    this.hadoopFileSystem.close();
  }
}

