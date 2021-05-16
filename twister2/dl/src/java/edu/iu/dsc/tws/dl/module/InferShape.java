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
package edu.iu.dsc.tws.dl.module;

import java.util.List;

import edu.iu.dsc.tws.dl.utils.Shape;
import edu.iu.dsc.tws.dl.utils.Util;

@SuppressWarnings("MemberName")
public abstract class InferShape {
  private Shape _inputShapeValue = null;

  private Shape _outputShapeValue = null;

  private Shape inputShapeValue = _inputShapeValue;

  private Shape outputShapeValue = _outputShapeValue;

  // scalastyle:off
  private void inputShapeValue_(Shape value) {
    _inputShapeValue = value;
  }

  private void outputShapeValue_(Shape value) {
    _outputShapeValue = value;
  }
  // scalastyle:on

  /**
   * Return the inputShape for the current Layer and the first dim is batch.
   */
  final Shape getInputShape() {
    Util.require(this.isKerasStyle(),
        "Torch style definition doesn't support getInputShape for now.");
    return _inputShapeValue;
  }

  private boolean isKerasStyle() {
    return false;
  }

  /**
   * Return the outputShape for the current Layer and the first dim is batch.
   */
  final Shape getOutputShape() {
    Util.require(this.isKerasStyle(),
        "Torch style definition doesn't support getOutputShape for now.");
    Util.require(this.isBuilt(), "This module hasn't been built.");
    return outputShapeValue;
  }

  /**
   * Execute building logic and return the outputShape for the given inputShape.
   * NB: the first dim of inputShape is batch
   */
  private Shape build(Shape inputShape) {
    Shape outputShape = computeOutputShape(inputShape);
    this.outputShapeValue = outputShape;
    this.inputShapeValue = inputShape;
    return outputShape;
  }

  private boolean isBuilt() {
    return outputShapeValue != null;
  }

  private boolean allowRebuilt() {
    return false;
  }

  /**
   * We suppose the first dim is batch
   */
  public Shape computeOutputShape(Shape inputShape) {
    throw new RuntimeException("Haven't been implemented yet. Do not use it with Keras Layer");
  }

  private void excludeInvalidLayers(List<AbstractModule> modules) {
    //TODO: implement after module is implemented
//    boolean invalidNodes;
//    if (this.isKerasStyle()) {
//      invalidNodes = modules.stream().filter(abstractModule -> abstractModule)
//    } else {
//      modules.filter{_.isKerasStyle()}
//    }
//    if (invalidNodes.length > 0) {
//      throw new InvalidLayer(s"""Do not mix ${this}(isKerasStyle=${isKerasStyle()}) with Layer
//                           (isKerasStyle=${invalidNodes(0).isKerasStyle()}):
//         ${invalidNodes.mkString(",")}""")
//    }
  }

  protected void validateInput(List<AbstractModule> modules) {
    if (this.isKerasStyle()) {
      Util.require(modules != null && !modules.isEmpty(), "Empty input is not allowed");
    }
    excludeInvalidLayers(modules);
  }
}

