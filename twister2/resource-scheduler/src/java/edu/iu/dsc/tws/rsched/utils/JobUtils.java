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
package edu.iu.dsc.tws.rsched.utils;

import java.io.File;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;
import java.util.logging.Logger;

import com.google.protobuf.ByteString;
import com.google.protobuf.InvalidProtocolBufferException;

import edu.iu.dsc.tws.api.config.Config;
import edu.iu.dsc.tws.api.config.Context;
import edu.iu.dsc.tws.api.scheduler.SchedulerContext;
import edu.iu.dsc.tws.api.util.KryoSerializer;
import edu.iu.dsc.tws.proto.system.job.JobAPI;
import edu.iu.dsc.tws.proto.utils.ComputeResourceUtils;

public final class JobUtils {
  private static final Logger LOG = Logger.getLogger(JobUtils.class.getName());

  private JobUtils() {
  }

  /**
   * Write the job file
   */
  public static boolean writeJobFile(JobAPI.Job job, String fileName) {
    // lets write a job file
    byte[] jobBytes = job.toByteArray();
    return FileUtils.writeToFile(fileName, jobBytes, true);
  }

  /**
   * Read the job file
   */
  public static JobAPI.Job readJobFile(Config cfg, String fileName) {
    try {
      byte[] fileBytes = FileUtils.readFromFile(fileName);
      JobAPI.Job.Builder builder = JobAPI.Job.newBuilder();

      return builder.mergeFrom(fileBytes).build();
    } catch (InvalidProtocolBufferException e) {
      throw new RuntimeException("Failed to read the job file: " + fileName);
    }
  }

  public static Map<String, Object> readCommandLineOpts() {
    Map<String, Object> ret = new HashMap<>();
    String commandOptions = System.getenv("TWISTER2_OPTIONS");
    if (commandOptions != null) {
      String[] configs = commandOptions.split(",");
      for (String config : configs) {
        String[] options = config.split(":");
        if (options.length == 2) {
          ret.put(options[0], options[1]);
        }
      }
    }
    return ret;
  }

  public static String jobClassPath(Config cfg, JobAPI.Job job, String wd) {
    StringBuilder classPathBuilder = new StringBuilder();
    if (job.getJobFormat().getType() == JobAPI.JobFormatType.JAR
        || job.getJobFormat().getType() == JobAPI.JobFormatType.PYTHON) {
      classPathBuilder.append(
          Paths.get(wd, job.getJobId(), job.getJobFormat().getJobFile()).toString());
    } else {
      // now get the files
      File jobLib = Paths.get(wd, job.getJobId(), "lib").toFile();
      File[] listOfFiles = jobLib.listFiles();
      if (listOfFiles != null) {
        for (int i = 0; i < listOfFiles.length; i++) {
          if (listOfFiles[i].isFile()) {
            if (classPathBuilder.length() != 0) {
              classPathBuilder.append(":").append(
                  Paths.get(jobLib.getPath(), listOfFiles[i].getName()).toString());
            } else {
              classPathBuilder.append(Paths.get(
                  jobLib.getPath(), listOfFiles[i].getName()).toString());
            }
          }
        }
      }
    }
    return classPathBuilder.toString();
  }

  public static String systemClassPath(Config cfg) {
    String libDirectory = SchedulerContext.libDirectory(cfg);
    String libFile = Paths.get(libDirectory).toString();
    String classPath = "";
    File folder = new File(libFile);
    String libName = folder.getName();
    File[] listOfFiles = folder.listFiles();

    if (listOfFiles != null) {
      for (int i = 0; i < listOfFiles.length; i++) {
        if (listOfFiles[i].isFile()) {
          if (!"".equals(classPath)) {
            classPath += ":" + Paths.get(libDirectory, listOfFiles[i].getName()).toString();
          } else {
            classPath += Paths.get(libDirectory, listOfFiles[i].getName()).toString();
          }
        }
      }
    }
    return classPath;
  }

  /**
   * [Deprecated Function]
   **/
 /* public static Config overrideConfigs(JobAPI.Job job, Config config) {
    Config.Builder builder = Config.newBuilder().putAll(config);
    JobAPI.Config conf = job.getConfig();
    for (JobAPI.Config.KeyValue kv : conf.getKvsList()) {
      builder.put(kv.getKey(), kv.getValue());
    }
    return builder.build();
  }*/
  public static Config overrideConfigs(JobAPI.Job job, Config config) {
    Config.Builder builder = Config.newBuilder().putAll(config);
    JobAPI.Config conf = job.getConfig();
    Map<String, ByteString> configMapSerialized = conf.getConfigByteMapMap();
    for (Map.Entry<String, ByteString> e : configMapSerialized.entrySet()) {
      String key = e.getKey();
      byte[] bytes = e.getValue().toByteArray();
      Object object = new KryoSerializer().deserialize(bytes);
      builder.put(key, object);
    }
    return builder.build();
  }

  public static String getJobDescriptionFilePath(String workingDirectory,
                                                 String jobFileName, Config config) {
    return Paths.get(workingDirectory, jobFileName + ".job").toAbsolutePath().toString();
  }

  public static String getJobDescriptionFilePath(String jobFileName, Config config) {
    String home = Context.twister2Home(config);
    return Paths.get(home, jobFileName + ".job").toAbsolutePath().toString();
  }

  /**
   * write the values from Job object to config object
   */
  public static Config updateConfigs(JobAPI.Job job, Config config) {
    Config.Builder builder = Config.newBuilder().putAll(config);

    builder.put(Context.JOB_NAME, job.getJobName());

    builder.put(SchedulerContext.WORKER_CLASS, job.getWorkerClassName());
    builder.put(Context.TWISTER2_WORKER_INSTANCES, job.getNumberOfWorkers());
    builder.put(Context.JOB_ID, job.getJobId());

    return builder.build();
  }

  /**
   * return the ComputeResource with the given index
   * if not found, return null
   */
  public static JobAPI.ComputeResource getComputeResource(JobAPI.Job job, int index) {
    for (JobAPI.ComputeResource computeResource : job.getComputeResourceList()) {
      if (computeResource.getIndex() == index) {
        return computeResource;
      }
    }

    return null;
  }

  public static String toString(JobAPI.Job job) {
    String jobStr =
        String.format("[jobName=%s], [jobID=%s], \n[numberOfWorkers=%s], [workerClass=%s]",
        job.getJobName(), job.getJobId(), job.getNumberOfWorkers(), job.getWorkerClassName());

    for (JobAPI.ComputeResource cr : job.getComputeResourceList()) {
      jobStr += "\n" + ComputeResourceUtils.toString(cr);
    }

    return jobStr;
  }

  public static String createJobPackageFileName(String jobID) {
    return jobID + ".tar.gz";
  }

  /**
   * For the job to be scalable:
   *   Driver class shall be specified
   *   a scalable compute resource shall be given
   *   itshould not be an openMPI job
   *
   * @return
   */
  public static boolean isJobScalable(Config config, JobAPI.Job job) {

    // if Driver is not set, it means there is nothing to scale the job
    if (job.getDriverClassName().isEmpty()) {
      return false;
    }

    // if there is no scalable compute resource in the job, can not be scalable
    boolean computeResourceScalable =
        job.getComputeResource(job.getComputeResourceCount() - 1).getScalable();
    if (!computeResourceScalable) {
      return false;
    }

    // if it is an OpenMPI job, it is not scalable
    if (SchedulerContext.useOpenMPI(config)) {
      return false;
    }

    return true;
  }

}
