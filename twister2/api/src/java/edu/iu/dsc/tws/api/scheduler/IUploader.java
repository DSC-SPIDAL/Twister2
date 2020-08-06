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

package edu.iu.dsc.tws.api.scheduler;

import java.net.URI;

import edu.iu.dsc.tws.api.config.Config;

/**
 * Uploads job package to a shared location.
 * This location must be accessible by Twister2 workers and the job master.
 * The uploader will upload:
 * <p>
 * - job package: including user job file, config files, job definition file, and
 * - twister2 core packages and libraries, if required
 * <p>
 * Implementation of IUploader is required to have a no argument constructor
 * that will be called to create an instance of IUploader.
 *
 * initialize method must be called after the constructor
 *
 */
public interface IUploader extends AutoCloseable {
  /**
   * Initialize the uploader with the incoming context.
   */
  void initialize(Config config, String jobID);

  /**
   * UploadPackage will upload the job package to the given location.
   *
   * @param sourceLocation the source location with all the files
   * @return destination URI of where the job package has
   * been uploaded if successful, or {@code null} if failed.
   */
  URI uploadPackage(String sourceLocation) throws UploaderException;

  /**
   * if uploader is threaded,
   * this method will wait for the threaded uploading to finish
   * @return
   */
  default boolean complete() {
    return true;
  }

  /**
   * Undo uploading. Remove uploaded files.
   */
  boolean undo();

  /**
   * This is to for disposing or cleaning up any internal state accumulated by the uploader
   */
  void close();
}
