"""
Fit step for gross alignment and scale.
"""
import copy
import math

from opencmiss.maths.vectorops import div, euler_to_rotation_matrix, matrix_vector_mult, mult, sub
from opencmiss.utils.zinc.field import get_group_list, create_field_euler_angles_rotation_matrix
from opencmiss.utils.zinc.finiteelement import evaluate_field_nodeset_mean, getNodeNameCentres
from opencmiss.utils.zinc.general import ChangeManager
from opencmiss.zinc.element import Mesh
from opencmiss.zinc.field import Field
from opencmiss.zinc.optimisation import Optimisation
from opencmiss.zinc.result import RESULT_OK, RESULT_WARNING_PART_DONE
from scaffoldfitter.fitterstep import FitterStep


def createFieldsTransformations(coordinates: Field, rotation_angles=None, scale_value=1.0,
                                translation_offsets=None, translation_scale_factor=1.0):
    """
    Create constant fields for rotation, scale and translation containing the supplied
    values, plus the transformed coordinates applying them in the supplied order.
    :param coordinates: The coordinate field to scale, 3 components.
    :param rotation_angles: List of euler angles, length = number of components.
     See create_field_euler_angles_rotation_matrix.
    :param scale_value: Scalar to multiply all components of coordinates.
    :param translation_offsets: List of offsets, length = number of components.
    :param translation_scale_factor: Scaling to multiply translation by so it's magnitude can remain
    close to other parameters for rotation (radians) and scale (assumed close to unit).
    :return: 4 fields: transformedCoordinates, rotation, scale, translation
    """
    if rotation_angles is None:
        rotation_angles = [0.0, 0.0, 0.0]
    if translation_offsets is None:
        translation_offsets = [0.0, 0.0, 0.0]
    components_count = coordinates.getNumberOfComponents()
    assert (components_count == 3) and (len(rotation_angles) == components_count) and isinstance(scale_value, float) \
           and (len(translation_offsets) == components_count), "createFieldsTransformations.  Invalid arguments"
    fieldmodule = coordinates.getFieldmodule()
    with ChangeManager(fieldmodule):
        # Rotate, scale, and translate model, in that order
        rotation = fieldmodule.createFieldConstant(rotation_angles)
        scale = fieldmodule.createFieldConstant(scale_value)
        translation = fieldmodule.createFieldConstant(translation_offsets)
        rotation_matrix = create_field_euler_angles_rotation_matrix(fieldmodule, rotation)
        rotated_coordinates = fieldmodule.createFieldMatrixMultiply(components_count, rotation_matrix, coordinates)
        transformed_coordinates = rotated_coordinates * scale + (
            translation if (translation_scale_factor == 1.0) else
            translation * fieldmodule.createFieldConstant([translation_scale_factor] * components_count))
        assert transformed_coordinates.isValid()
    return transformed_coordinates, rotation, scale, translation


class FitterStepAlign(FitterStep):

    _jsonTypeId = "_FitterStepAlign"

    def __init__(self):
        super(FitterStepAlign, self).__init__()
        self._alignGroups = False
        self._alignMarkers = False
        self._rotation = [0.0, 0.0, 0.0]
        self._scale = 1.0
        self._translation = [0.0, 0.0, 0.0]

    @classmethod
    def getJsonTypeId(cls):
        return cls._jsonTypeId

    def decodeSettingsJSONDict(self, dctIn: dict):
        """
        Decode definition of step from JSON dict.
        """
        super().decodeSettingsJSONDict(dctIn)  # to decode group settings
        # ensure all new options are in dct
        dct = self.encodeSettingsJSONDict()
        dct.update(dctIn)
        self._alignGroups = dct["alignGroups"]
        self._alignMarkers = dct["alignMarkers"]
        self._rotation = dct["rotation"]
        self._scale = dct["scale"]
        self._translation = dct["translation"]

    def encodeSettingsJSONDict(self) -> dict:
        """
        Encode definition of step in dict.
        :return: Settings in a dict ready for passing to json.dump.
        """
        dct = super().encodeSettingsJSONDict()
        dct.update({
            "alignGroups": self._alignGroups,
            "alignMarkers": self._alignMarkers,
            "rotation": self._rotation,
            "scale": self._scale,
            "translation": self._translation
        })

        return dct

    def isAlignGroups(self):
        return self._alignGroups

    def setAlignGroups(self, alignGroups):
        """
        Set whether alignment includes mapping mean position of annotation
        groups in model to mean position of data for that group.
        :param alignGroups: True to automatically align to groups, otherwise False.
        :return: True if state changed, otherwise False.
        """
        if alignGroups != self._alignGroups:
            self._alignGroups = alignGroups
            return True
        return False

    def isAlignMarkers(self):
        return self._alignMarkers

    def setAlignMarkers(self, alignMarkers):
        """
        Set whether alignment includes mapping model marker positions
        to data marker positions.
        :param alignMarkers: True to automatically align to markers, otherwise False.
        :return: True if state changed, otherwise False.
        """
        if alignMarkers != self._alignMarkers:
            self._alignMarkers = alignMarkers
            return True
        return False

    def getRotation(self):
        return self._rotation

    def setRotation(self, rotation):
        """
        :param rotation: List of 3 euler angles in radians, order applied:
        0 = azimuth (about z)
        1 = elevation (about rotated y)
        2 = roll (about rotated x)
        :return: True if state changed, otherwise False.
        """
        assert len(rotation) == 3, "FitterStepAlign:  Invalid rotation"
        if rotation != self._rotation:
            self._rotation = copy.copy(rotation)
            return True
        return False

    def getScale(self):
        return self._scale

    def setScale(self, scale):
        """
        :param scale: Real scale.
        :return: True if state changed, otherwise False.
        """
        if scale != self._scale:
            self._scale = scale
            return True
        return False

    def getTranslation(self):
        return self._translation

    def setTranslation(self, translation):
        """
        :param translation: [ x, y, z ].
        :return: True if state changed, otherwise False.
        """
        assert len(translation) == 3, "FitterStepAlign:  Invalid translation"
        if translation != self._translation:
            self._translation = copy.copy(translation)
            return True
        return False

    def run(self, modelFileNameStem=None):
        """
        Perform align and scale.
        :param modelFileNameStem: Optional name stem of intermediate output file to write.
        """
        modelCoordinates = self._fitter.getModelCoordinatesField()
        assert modelCoordinates, "Align:  Missing model coordinates"
        if self._alignGroups or self._alignMarkers:
            self._doAutoAlign()
        fieldmodule = self._fitter.getFieldmodule()
        with ChangeManager(fieldmodule):
            # rotate, scale and translate model
            modelCoordinatesTransformed = createFieldsTransformations(
                modelCoordinates, self._rotation, self._scale, self._translation)[0]
            fieldassignment = self._fitter.getModelCoordinatesField().createFieldassignment(modelCoordinatesTransformed)
            result = fieldassignment.assign()
            assert result in [RESULT_OK, RESULT_WARNING_PART_DONE], "Align:  Failed to transform model"
            self._fitter.updateModelReferenceCoordinates()
            del fieldassignment
            del modelCoordinatesTransformed
        self._fitter.calculateDataProjections(self)
        if modelFileNameStem:
            self._fitter.writeModel(modelFileNameStem + "_align.exf")
        self.setHasRun(True)

    def _doAutoAlign(self):
        """
        Perform auto alignment to groups and/or markers.
        """
        modelCoordinates = self._fitter.getModelCoordinatesField()
        pointMap = {}  # dict group/marker name -> (modelCoordinates, dataCoordinates)

        if self._alignGroups:
            fieldmodule = self._fitter.getFieldmodule()
            dataCoordinates = self._fitter.getDataCoordinatesField()
            groups = get_group_list(fieldmodule)
            with ChangeManager(fieldmodule):
                one = fieldmodule.createFieldConstant(1.0)
                for group in groups:
                    dataGroup = self._fitter.getGroupDataProjectionNodesetGroup(group)
                    if not dataGroup:
                        continue
                    meshGroup = self._fitter.getGroupDataProjectionMeshGroup(group)
                    if not meshGroup:
                        continue
                    groupName = group.getName()
                    meanDataCoordinates = evaluate_field_nodeset_mean(dataCoordinates, dataGroup)
                    coordinates_integral = evaluate_field_mesh_integral(modelCoordinates, modelCoordinates, meshGroup)
                    mass = evaluate_field_mesh_integral(one, modelCoordinates, meshGroup)
                    meanModelCoordinates = div(coordinates_integral, mass)
                    pointMap[groupName] = (meanModelCoordinates, meanDataCoordinates)
                del one

        if self._alignMarkers:
            markerGroup = self._fitter.getMarkerGroup()
            assert markerGroup, "Align:  No marker group to align with"

            markerNodeGroup, markerLocation, markerCoordinates, markerName = self._fitter.getMarkerModelFields()
            assert markerNodeGroup and markerCoordinates and markerName, \
                "Align:  No marker group, coordinates or name fields"
            modelMarkers = getNodeNameCentres(markerNodeGroup, markerCoordinates, markerName)

            markerDataGroup, markerDataCoordinates, markerDataName = self._fitter.getMarkerDataFields()
            assert markerDataGroup and markerDataCoordinates and markerDataName, \
                "Align:  No marker data group, coordinates or name fields"
            dataMarkers = getNodeNameCentres(markerDataGroup, markerDataCoordinates, markerDataName)

            # match model and data markers, warn of unmatched markers
            writeDiagnostics = self.getDiagnosticLevel() > 0
            for modelName in modelMarkers:
                # name match allows case and whitespace differences
                matchName = modelName.strip().casefold()
                for dataName in dataMarkers:
                    if dataName.strip().casefold() == matchName:
                        pointMap[modelName] = (modelMarkers[modelName], dataMarkers[dataName])
                        if writeDiagnostics:
                            print("Align:  Model marker '" + modelName + "' found in data" +
                                  (" as '" + dataName + "'" if (dataName != modelName) else ""))
                            dataMarkers.pop(dataName)
                        break
                else:
                    if writeDiagnostics:
                        print("Align:  Model marker '" + modelName + "' not found in data")
            if writeDiagnostics:
                for dataName in dataMarkers:
                    print("Align:  Data marker '" + dataName + "' not found in model")

        self._optimiseAlignment(pointMap)

    def _optimiseAlignment(self, pointMap):
        """
        Calculate transformation from modelCoordinates to dataMarkers
        over the markers, by scaling, translating and rotating model.
        On success, sets transformation parameters in object.
        :param pointMap: dict name -> (modelCoordinates, dataCoordinates)
        """
        assert len(pointMap) >= 3, "Align:  Only " + str(len(pointMap)) + " points - need at least 3"
        region = self._fitter.getContext().createRegion()
        fieldmodule = region.getFieldmodule()

        with ChangeManager(fieldmodule):
            modelCoordinates = fieldmodule.createFieldFiniteElement(3)
            dataCoordinates = fieldmodule.createFieldFiniteElement(3)
            nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
            nodetemplate = nodes.createNodetemplate()
            nodetemplate.defineField(modelCoordinates)
            nodetemplate.defineField(dataCoordinates)
            fieldcache = fieldmodule.createFieldcache()

            modelsum = [0.0, 0.0, 0.0]
            datasum = [0.0, 0.0, 0.0]
            modelMin = copy.deepcopy(list(pointMap.values())[0][0])
            modelMax = copy.deepcopy(list(pointMap.values())[0][0])
            dataMin = copy.deepcopy(list(pointMap.values())[0][1])
            dataMax = copy.deepcopy(list(pointMap.values())[0][1])
            for name, positions in pointMap.items():
                modelx = positions[0]
                datax = positions[1]
                modelsum = [(modelsum[c] + modelx[c]) for c in range(3)]
                datasum = [(datasum[c] + datax[c]) for c in range(3)]
                node = nodes.createNode(-1, nodetemplate)
                fieldcache.setNode(node)
                result1 = modelCoordinates.assignReal(fieldcache, positions[0])
                result2 = dataCoordinates.assignReal(fieldcache, positions[1])
                assert (result1 == RESULT_OK) and (result2 == RESULT_OK), \
                    "Align:  Failed to set up data for alignment to markers optimisation"
                for c in range(3):
                    modelMin[c] = min(modelx[c], modelMin[c])
                    modelMax[c] = max(modelx[c], modelMax[c])
                    dataMin[c] = min(datax[c], dataMin[c])
                    dataMax[c] = max(datax[c], dataMax[c])

            groupScale = 1.0 / len(pointMap)
            modelCM = [(s * groupScale) for s in modelsum]
            dataCM = [(s * groupScale) for s in datasum]
            translationOffset = [(dataCM[c] - modelCM[c]) for c in range(3)]

            modelSpan = [(modelMax[c] - modelMin[c]) for c in range(3)]
            dataSpan = [(dataMax[c] - dataMin[c]) for c in range(3)]
            scaleFactor = max(dataSpan) / max(modelSpan)

            del fieldcache

            # Pre-align to avoid gimbal lock
            translationScaleFactor = 1.0
            first = True
            fieldcache = fieldmodule.createFieldcache()
            modelCoordinatesTransformed, rotation, scale, translation = \
                createFieldsTransformations(modelCoordinates, scale_value=scaleFactor)

            # create objective = sum of squares of vector from modelCoordinatesTransformed to dataCoordinates
            markerDiff = fieldmodule.createFieldSubtract(dataCoordinates, modelCoordinatesTransformed)
            scaledMarkerDiff = markerDiff * fieldmodule.createFieldConstant([1.0 / translationScaleFactor] * 3)
            objective = fieldmodule.createFieldNodesetSumSquares(scaledMarkerDiff, nodes)

            for x in range(2):
                for y in (range(4) if x == 0 else (0, 2)):
                    for z in range(4):
                        azimuth = 0.5 * math.pi * z
                        elevation = 0.5 * math.pi * y
                        roll = 0.5 * math.pi * x
                        rotationAngles = [azimuth, elevation, roll]
                        rotation.assignReal(fieldcache, rotationAngles)
                        rotationMatrix = euler_to_rotation_matrix(rotationAngles)
                        translationFix = mult(sub(matrix_vector_mult(rotationMatrix, modelCM), modelCM), scaleFactor)
                        translation.assignReal(fieldcache, sub(translationOffset, translationFix))
                        result, objectiveValues = objective.evaluateReal(fieldcache, 3)
                        objectiveValue = sum(objectiveValues)

                        if first or (objectiveValue < minObjectiveValue):
                            first = False
                            minObjectiveValue = objectiveValue
                            minRotationAngles = rotationAngles
                            minTranslation = sub(translationOffset, translationFix)

            rotation.assignReal(fieldcache, minRotationAngles)
            translation.assignReal(fieldcache, minTranslation)

            assert objective.isValid(), \
                "Align:  Failed to set up objective function for alignment to markers optimisation"

        optimisation = fieldmodule.createOptimisation()
        optimisation.setMethod(Optimisation.METHOD_LEAST_SQUARES_QUASI_NEWTON)
        # optimisation.setMethod(Optimisation.METHOD_QUASI_NEWTON)
        optimisation.addObjectiveField(objective)
        optimisation.addDependentField(rotation)
        optimisation.addDependentField(scale)
        optimisation.addDependentField(translation)

        # FunctionTolerance = optimisation.getAttributeReal(Optimisation.ATTRIBUTE_FUNCTION_TOLERANCE)
        # GradientTolerance = optimisation.getAttributeReal(Optimisation.ATTRIBUTE_GRADIENT_TOLERANCE)
        # StepTolerance = optimisation.getAttributeReal(Optimisation.ATTRIBUTE_STEP_TOLERANCE)
        # MaximumStep = optimisation.getAttributeReal(Optimisation.ATTRIBUTE_MAXIMUM_STEP)
        # MinimumStep = optimisation.getAttributeReal(Optimisation.ATTRIBUTE_MINIMUM_STEP)
        # LinesearchTolerance = optimisation.getAttributeReal(Optimisation.ATTRIBUTE_LINESEARCH_TOLERANCE)
        # TrustRegionSize = optimisation.getAttributeReal(Optimisation.ATTRIBUTE_TRUST_REGION_SIZE)

        # tol_scale = dataScale*dataScale
        # FunctionTolerance *= tol_scale
        # optimisation.setAttributeReal(Optimisation.ATTRIBUTE_FUNCTION_TOLERANCE, FunctionTolerance)
        # GradientTolerance *= tol_scale
        # optimisation.setAttributeReal(Optimisation.ATTRIBUTE_GRADIENT_TOLERANCE, GradientTolerance)
        # StepTolerance *= tol_scale
        # optimisation.setAttributeReal(Optimisation.ATTRIBUTE_STEP_TOLERANCE, StepTolerance)
        # MaximumStep *= tol_scale
        # optimisation.setAttributeReal(Optimisation.ATTRIBUTE_MAXIMUM_STEP, MaximumStep)
        # MinimumStep *= tol_scale
        # optimisation.setAttributeReal(Optimisation.ATTRIBUTE_MINIMUM_STEP, MinimumStep)
        # LinesearchTolerance *= dataScale
        # optimisation.setAttributeReal(Optimisation.ATTRIBUTE_LINESEARCH_TOLERANCE, LinesearchTolerance)
        # TrustRegionSize *= dataScale
        # optimisation.setAttributeReal(Optimisation.ATTRIBUTE_TRUST_REGION_SIZE, TrustRegionSize)

        # if self.getDiagnosticLevel() > 0:
        #    print("Function Tolerance", FunctionTolerance)
        #    print("Gradient Tolerance", GradientTolerance)
        #    print("Step Tolerance", StepTolerance)
        #    print("Maximum Step", MaximumStep)
        #    print("Minimum Step", MinimumStep)
        #    print("Linesearch Tolerance", LinesearchTolerance)
        #    print("Trust Region Size", TrustRegionSize)

        result = optimisation.optimise()
        if self.getDiagnosticLevel() > 1:
            solutionReport = optimisation.getSolutionReport()
            print(solutionReport)
        assert result == RESULT_OK, "Align:  Alignment to markers optimisation failed"

        result1, self._rotation = rotation.evaluateReal(fieldcache, 3)
        result2, self._scale = scale.evaluateReal(fieldcache, 1)
        result3, self._translation = translation.evaluateReal(fieldcache, 3)
        self._translation = [s * translationScaleFactor for s in self._translation]
        assert (result1 == RESULT_OK) and (result2 == RESULT_OK) and (result3 == RESULT_OK), \
            "Align:  Failed to evaluate transformation for alignment to markers"


def evaluate_field_mesh_integral(field: Field, coordinates: Field, mesh: Mesh, number_of_points=4):
    """
    Integrate value of a field over mesh using Gaussian Quadrature.
    :param field: Field to integrate over mesh.
    :param coordinates: Field giving spatial coordinates to integrate over.
    :param mesh: The mesh or mesh group to integrate over.
    :param number_of_points: Number of integration points in each dimension.
    :return: Integral value.
    """
    fieldmodule = mesh.getFieldmodule()
    components_count = field.getNumberOfComponents()
    with ChangeManager(fieldmodule):
        integral = fieldmodule.createFieldMeshIntegral(field, coordinates, mesh)
        integral.setNumbersOfPoints(number_of_points)
        fieldcache = fieldmodule.createFieldcache()
        result, value = integral.evaluateReal(fieldcache, components_count)
        del integral
        del fieldcache
    assert result == RESULT_OK
    return value
