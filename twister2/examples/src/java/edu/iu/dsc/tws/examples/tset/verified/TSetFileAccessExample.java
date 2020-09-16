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
package edu.iu.dsc.tws.examples.tset.verified;

import edu.iu.dsc.tws.api.resource.WorkerEnvironment;
import edu.iu.dsc.tws.examples.tset.BaseTSetBatchWorker;
import edu.iu.dsc.tws.tset.env.BatchEnvironment;
import edu.iu.dsc.tws.tset.env.TSetEnvironment;

public class TSetFileAccessExample extends BaseTSetBatchWorker {
  @Override
  public void execute(WorkerEnvironment workerEnv) {
    super.execute(workerEnv);
    BatchEnvironment env = TSetEnvironment.initBatch(workerEnv);
/*    super.execute(tc);

    String inputDirectory = config.getStringValue(Constants.ARGS_FNAME,
        "/tmp/twister2");
    int numFiles = config.getIntegerValue(Constants.ARGS_WORKERS, 4);
    int size = config.getIntegerValue(Constants.ARGS_SIZE, 1000);

    String input = inputDirectory + "/input";
    String output = inputDirectory + "/output";
    if (workerId == 0) {
      try {
        new File(input).mkdirs();
        new File(output).mkdirs();

        DataGenerator.generateData("txt", new Path(input),
            numFiles, size, 10);
      } catch (IOException e) {
        throw new RuntimeException("Failed to create data: " + input);
      }
    }

    BatchSourceTSet<String> textSource = tc.createSource(new FileSource<>(
        new SharedTextInputPartitioner(new Path(input))), jobParameters.getTaskStages().get(0));

    textSource.partition(new OneToOnePartitioner<>()).sink(
        new FileSink<>(new TextOutputWriter(
            FileSystem.WriteMode.OVERWRITE,
            new Path(output))), jobParameters.getTaskStages().get(0)); */
  }
}
